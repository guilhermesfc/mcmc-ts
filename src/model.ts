import { LogDensity, Vector } from "./core.js";
import {
  Transform,
  TransformSpec,
  positiveTransform,
  lowerBoundedTransform,
  upperBoundedTransform,
  boundedTransform,
  unitIntervalTransform,
  simplexTransform,
  composeTransforms,
  identityTransform,
} from "./transforms.js";
import { metropolisHastings, MHResult } from "./mh.js";
import { summarizeChains, ChainSummary } from "./diagnostics.js";

// Constraint specification types
export type ConstraintSpec =
  | "positive"
  | "unconstrained"
  | "unitInterval"
  | { type: "positive" }
  | { type: "unconstrained" }
  | { type: "unitInterval" }
  | { type: "bounded"; lower: number; upper: number }
  | { type: "lowerBounded"; lower: number }
  | { type: "upperBounded"; upper: number }
  | { type: "simplex"; k: number };

export interface ModelOptions {
  logDensity: LogDensity;
  dim: number;
  constraints?: Record<number, ConstraintSpec>;
  transforms?: TransformSpec[];
}

export interface SampleOptions {
  sampler?: "mh";
  chains?: number;
  iterations: number;
  warmup?: number;
  thin?: number;
  seed?: bigint;
  start?: Vector;
  chainStarts?: Vector[];
  stepSize?: number | number[];

  // Adaptation
  targetAcceptance?: number;
  adaptSteps?: number;
  adaptWindow?: number;

  // What to return
  includeWarmup?: boolean;
  includeUnconstrained?: boolean;
}

export interface SampleResult {
  // Core draws
  draws: Vector[][];
  warmupDraws: Vector[][];

  // Diagnostics
  acceptanceRates: number[];
  finalStepSizes: number[][];

  // Raw data
  rawTraces: Vector[][];
  unconstrainedDraws?: Vector[][];

  // Convenience method
  summary(): ChainSummary;
}

export interface Model {
  readonly constrainedDim: number;
  readonly unconstrainedDim: number;
  readonly transform: Transform;
  sample(opts: SampleOptions): SampleResult;
}

/**
 * Convert a constraint specification to a Transform
 */
function constraintToTransform(spec: ConstraintSpec): Transform {
  if (typeof spec === "string") {
    switch (spec) {
      case "positive":
        return positiveTransform(1);
      case "unconstrained":
        return identityTransform(1);
      case "unitInterval":
        return unitIntervalTransform(1);
      default:
        throw new Error(`Unknown constraint type: ${spec}`);
    }
  }

  switch (spec.type) {
    case "positive":
      return positiveTransform(1);
    case "unconstrained":
      return identityTransform(1);
    case "unitInterval":
      return unitIntervalTransform(1);
    case "bounded":
      return boundedTransform(spec.lower, spec.upper, 1);
    case "lowerBounded":
      return lowerBoundedTransform(spec.lower, 1);
    case "upperBounded":
      return upperBoundedTransform(spec.upper, 1);
    case "simplex":
      return simplexTransform(spec.k);
    default:
      throw new Error(`Unknown constraint type: ${(spec as { type: string }).type}`);
  }
}

/**
 * Convert constraints record to TransformSpec array
 */
function constraintsToTransforms(
  constraints: Record<number, ConstraintSpec>,
): TransformSpec[] {
  const specs: TransformSpec[] = [];

  for (const [indexStr, constraint] of Object.entries(constraints)) {
    const index = parseInt(indexStr, 10);
    if (isNaN(index)) {
      throw new Error(`Invalid constraint index: ${indexStr}`);
    }

    // Skip unconstrained (will be filled with identity by composeTransforms)
    if (constraint === "unconstrained" || (typeof constraint === "object" && constraint.type === "unconstrained")) {
      continue;
    }

    specs.push({
      startIndex: index,
      transform: constraintToTransform(constraint),
    });
  }

  return specs;
}

/**
 * Define a model with a log-density and optional constraints
 */
export function defineModel(options: ModelOptions): Model {
  const { logDensity, dim } = options;

  // Build transforms from constraints or use provided transforms
  let transformSpecs: TransformSpec[];
  if (options.constraints) {
    if (options.transforms) {
      throw new Error("Cannot specify both constraints and transforms");
    }
    transformSpecs = constraintsToTransforms(options.constraints);
  } else {
    transformSpecs = options.transforms ?? [];
  }

  // Compose all transforms
  const transform = composeTransforms(dim, transformSpecs);

  return {
    constrainedDim: dim,
    unconstrainedDim: transform.inputDim,
    transform,

    sample(sampleOpts: SampleOptions): SampleResult {
      const {
        sampler = "mh",
        chains = 1,
        iterations,
        warmup = 0,
        thin = 1,
        seed,
        start,
        chainStarts,
        stepSize,
        targetAcceptance,
        adaptSteps,
        adaptWindow,
        includeWarmup = false,
        includeUnconstrained = false,
      } = sampleOpts;

      if (sampler !== "mh") {
        throw new Error(`Unknown sampler: ${sampler}. Currently only 'mh' is supported.`);
      }

      // Run MH sampler
      // We need to handle warmup ourselves since we want to return warmup draws separately
      const totalIterations = iterations + warmup;

      const mhResult: MHResult = metropolisHastings(logDensity, dim, {
        iterations: totalIterations,
        burnIn: 0, // Don't burn in - we'll separate warmup ourselves
        thin,
        seed,
        start,
        chainStarts,
        chains,
        stepSize,
        transforms: transformSpecs,
        targetAcceptance,
        adaptSteps: adaptSteps ?? warmup, // Default: adapt during warmup
        adaptWindow,
      });

      // Separate warmup and post-warmup draws
      const warmupCount = Math.floor(warmup / thin);
      const warmupDraws: Vector[][] = [];
      const draws: Vector[][] = [];

      for (const chainSamples of mhResult.samples) {
        warmupDraws.push(chainSamples.slice(0, warmupCount));
        draws.push(chainSamples.slice(warmupCount));
      }

      // Get final step sizes (we don't have history yet, so just return the configured one)
      const finalStepSizes: number[][] = [];
      const stepSizeArray = Array.isArray(stepSize)
        ? stepSize
        : Array(transform.inputDim).fill(stepSize ?? 0.5);

      for (let c = 0; c < chains; c++) {
        finalStepSizes.push(stepSizeArray.slice());
      }

      // Build result
      const result: SampleResult = {
        draws,
        warmupDraws: includeWarmup ? warmupDraws : [],
        acceptanceRates: mhResult.acceptanceRates,
        finalStepSizes,
        rawTraces: mhResult.rawTraces,

        summary(): ChainSummary {
          return summarizeChains(draws);
        },
      };

      if (includeUnconstrained) {
        // Transform draws back to unconstrained space
        result.unconstrainedDraws = draws.map((chain) =>
          chain.map((x) => transform.inverse(x)),
        );
      }

      return result;
    },
  };
}
