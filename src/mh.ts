import { LogDensity, PCG32, Vector, add, zeros } from "./core.js";
import {
  TransformSpec,
  transformedLogDensity,
  applyTransformToChain,
  composeTransforms,
  identityTransform,
} from "./transforms.js";

export interface MHOptions {
  stepSize?: number | number[]; // proposal std dev (scalar or per-dimension)
  burnIn?: number; // how many inital samples to discard
  thin?: number; // keep 1 of every 'thin' samples
  iterations: number; // total proposals
  seed?: bigint; // RNG seed for reproducibility (default: 0)
  start?: Vector; // initial point (for single chain, in constrained space if transforms provided)
  chains?: number; // number of chains to run (default: 1)
  chainStarts?: Vector[]; // custom starting points per chain (in constrained space if transforms provided)
  transforms?: TransformSpec[]; // optional block transforms (sparse, identity-filled)
  targetAcceptance?: number; // target acceptance rate for adaptation (e.g. 0.23)
  adaptSteps?: number; // number of iterations to adapt step size (only during warmup)
  adaptWindow?: number; // window size for computing acceptance rate (default: 50)
}

export interface MHResult {
  samples: Vector[][]; // [chain][draw] (always 2D, even for chains=1)
  acceptanceRates: number[]; // [chain]
  rawTraces: Vector[][]; // [chain][draw]
}

export function metropolisHastings(
  logDensity: LogDensity,
  dim: number,
  opts: MHOptions,
): MHResult {
  const {
    stepSize = 0.5,
    burnIn = 0,
    thin = 1,
    iterations,
    start = zeros(dim),
    chains = 1,
    chainStarts,
    seed = 0n,
    transforms,
    targetAcceptance,
    adaptSteps = 0,
    adaptWindow = 50,
  } = opts;

  const userProvidedStart = opts.start !== undefined;

  // Apply transformation (identity if none provided)
  const transform = transforms
    ? composeTransforms(dim, transforms)
    : identityTransform(dim);

  if (transform.outputDim !== dim) {
    throw new Error(
      `Transform outputDim (${transform.outputDim}) must equal dim (${dim})`,
    );
  }

  dim = transform.inputDim;
  logDensity = transformedLogDensity(logDensity, transform);

  // If user provided start, transform starting points to unconstrained space
  const actualStart = userProvidedStart ? transform.inverse(start) : zeros(dim);
  const actualChainStarts = chainStarts?.map((s) => transform.inverse(s));

  // Normalize stepSize to array (in unconstrained space)
  const stepSizeArray: number[] = Array.isArray(stepSize)
    ? stepSize
    : Array(dim).fill(stepSize);

  if (stepSizeArray.length !== dim) {
    throw new Error(
      `stepSize array length (${stepSizeArray.length}) must equal unconstrained dim (${dim})`,
    );
  }

  // Generate starting points for each chain
  const starts: Vector[] = actualChainStarts
    ? actualChainStarts
    : chains === 1
      ? [actualStart]
      : generateDispersedStarts(dim, chains, seed);

  if (starts.length !== chains) {
    throw new Error(
      `chainStarts length (${starts.length}) must equal chains (${chains})`,
    );
  }

  const allSamples: Vector[][] = [];
  const allRawTraces: Vector[][] = [];
  const acceptanceRates: number[] = [];

  // Run each chain independently
  for (let c = 0; c < chains; c++) {
    // Create independent RNG for each chain with deterministic seed
    // Chain c gets seed: (seed + c, c) for reproducibility
    const chainRng = new PCG32(seed + BigInt(c), BigInt(c));

    let x: Vector = starts[c].slice();
    let logp = logDensity(x);
    const rawTrace: Vector[] = [x.slice()];
    let accepted = 0;

    // Adaptive step size state (per chain, per dimension)
    const logStepArray = stepSizeArray.map((s) => Math.log(s));
    const currentStepArray = stepSizeArray.slice();
    let acceptedInWindow = 0;
    const doAdapt = targetAcceptance != null && adaptSteps > 0;

    for (let t = 0; t < iterations; t++) {
      // propose x' = x + N(0, currentStepArray[i]^2) per dimension
      const proposal: Vector = x.map((_, i) =>
        chainRng.normal(0, currentStepArray[i]),
      );
      const xNew = add(x, proposal);
      const logpNew = logDensity(xNew);

      // symmetric proposal => MH ratio = exp(logpNew - logp)
      const logAlpha = logpNew - logp;
      const accept = Math.log(chainRng.uniform()) < logAlpha;

      if (accept) {
        x = xNew;
        logp = logpNew;
        accepted++;
        acceptedInWindow++;
      }
      rawTrace.push(x.slice());

      // Adaptive step size update (Robbins–Monro algorithm)
      // Scales all dimensions proportionally
      if (doAdapt && t < adaptSteps && (t + 1) % adaptWindow === 0) {
        const accRate = acceptedInWindow / adaptWindow;
        const delta = accRate - targetAcceptance!;
        const eta = 0.05;
        for (let i = 0; i < dim; i++) {
          logStepArray[i] += eta * delta;
          currentStepArray[i] = Math.exp(logStepArray[i]);
        }
        acceptedInWindow = 0;
      }
    }

    // burn-in + thinning
    const kept: Vector[] = [];
    for (let i = burnIn; i < rawTrace.length; i += thin) {
      kept.push(rawTrace[i]);
    }

    allSamples.push(kept);
    allRawTraces.push(rawTrace);
    acceptanceRates.push(accepted / iterations);
  }

  // Transform samples back to constrained space
  return {
    samples: allSamples.map((chain) => applyTransformToChain(chain, transform)),
    acceptanceRates,
    rawTraces: allRawTraces.map((chain) =>
      applyTransformToChain(chain, transform),
    ),
  };
}

// Following Stan's default initialization strategy: Uniform(-2, 2) on unconstrained scale.
function generateDispersedStarts(
  dim: number,
  nChains: number,
  seed: bigint,
): Vector[] {
  // Use a different seed for generating starting points (seed + 1000000)
  // to avoid correlation with chain RNG seeds
  const rng = new PCG32(seed + 1000000n, 0n);
  const starts: Vector[] = [];

  for (let c = 0; c < nChains; c++) {
    // Generate starts uniformly from [-2, 2] for each dimension (matches Stan default)
    const start = Array(dim)
      .fill(0)
      .map(() => rng.uniform() * 4 - 2);
    starts.push(start);
  }

  return starts;
}
