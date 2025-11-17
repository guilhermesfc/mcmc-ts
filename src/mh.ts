import { LogDensity, RNG, PCG32, Vector, add, scale, zeros } from "./core.js";

export interface MHOptions {
  stepSize?: number; // proposal std dev (isotropic)
  burnIn?: number; // how many inital samples to discard when returning
  thin?: number; // keep 1 of every 'thin' samples
  iterations: number; // total proposals
  rng?: RNG;
  start?: Vector; // initial point
}

export interface MHResult {
  chain: Vector[]; // post burn-in/thinning
  acceptanceRate: number; // accepted / interations
  rawTrace: Vector[]; // full raw trace (including burn-in)
}

export function metropolisHastings(
  logDensity: LogDensity,
  dim: number,
  opts: MHOptions
): MHResult {
  const {
    stepSize = 0.5,
    burnIn = 0,
    thin = 1,
    iterations,
    rng = new PCG32(),
    start = zeros(dim),
  } = opts;

  let x: Vector = start.slice();
  let logp = logDensity(x);
  const rawTrace: Vector[] = [x.slice()];
  let accepted = 0;

  for (let t = 0; t < iterations; t++) {
    // propose x' = x + N(0, stepSize^2 I)
    const proposal: Vector = x.map(() => rng.normal(0, stepSize));
    const xNew = add(x, proposal);
    const logpNew = logDensity(xNew);

    // symmetric proposal => MH ratio = exp(logpNew - logp)
    const logAlpha = logpNew - logp;
    const accept = Math.log(rng.uniform()) < logAlpha;

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

  return {
    chain: kept,
    acceptanceRate: accepted / iterations,
    rawTrace,
  };
}
