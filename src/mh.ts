import { LogDensity, PCG32, Vector, add, zeros } from "./core.js";
import {
  Transform,
  transformedLogDensity,
  applyTransformToChain,
  composeTransforms,
} from "./transforms.js";

export interface MHOptions {
  stepSize?: number; // proposal std dev (isotropic)
  burnIn?: number; // how many inital samples to discard when returning
  thin?: number; // keep 1 of every 'thin' samples
  iterations: number; // total proposals
  seed?: bigint; // RNG seed for reproducibility (default: 0)
  start?: Vector; // initial point (for single chain, in constrained space if transforms provided)
  chains?: number; // number of chains to run (default: 1)
  chainStarts?: Vector[]; // custom starting points per chain (in constrained space if transforms provided)
  transforms?: Transform[]; // optional transforms (one per dimension)
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
  } = opts;

  // Validate transforms (if provided)
  if (transforms) {
    if (transforms.length !== dim) {
      throw new Error(
        `transforms.length (${transforms.length}) must equal dim (${dim})`,
      );
    }
  }

  // Compose transforms and prepare transformed log density
  let composedTransform: Transform | undefined;
  let actualLogDensity = logDensity;

  if (transforms) {
    composedTransform = composeTransforms(transforms);
    actualLogDensity = transformedLogDensity(logDensity, composedTransform);
  }

  // Transform user-provided starting points (if transforms are used)
  let actualStart = start;
  let actualChainStarts = chainStarts;

  if (transforms && composedTransform) {
    // User provides constrained starts, we need unconstrained
    if (start && start !== zeros(dim)) {
      actualStart = composedTransform.inverse(start);
    }
    if (chainStarts) {
      actualChainStarts = chainStarts.map((s) => composedTransform!.inverse(s));
    }
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
    let logp = actualLogDensity(x);
    const rawTrace: Vector[] = [x.slice()];
    let accepted = 0;

    for (let t = 0; t < iterations; t++) {
      // propose x' = x + N(0, stepSize^2 I)
      const proposal: Vector = x.map(() => chainRng.normal(0, stepSize));
      const xNew = add(x, proposal);
      const logpNew = actualLogDensity(xNew);

      // symmetric proposal => MH ratio = exp(logpNew - logp)
      const logAlpha = logpNew - logp;
      const accept = Math.log(chainRng.uniform()) < logAlpha;

      if (accept) {
        x = xNew;
        logp = logpNew;
        accepted++;
      }
      rawTrace.push(x.slice());
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

  // Transform samples back to constrained space (if transforms were used)
  if (transforms && composedTransform) {
    const constrainedSamples = allSamples.map((chain) =>
      applyTransformToChain(chain, composedTransform!),
    );
    const constrainedRawTraces = allRawTraces.map((chain) =>
      applyTransformToChain(chain, composedTransform!),
    );
    return {
      samples: constrainedSamples,
      acceptanceRates,
      rawTraces: constrainedRawTraces,
    };
  }

  return {
    samples: allSamples,
    acceptanceRates,
    rawTraces: allRawTraces,
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
