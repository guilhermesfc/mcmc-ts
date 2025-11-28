import { Vector } from "./core.js";

// Univariate ESS (very rough): batch means or simple autocorr truncation.
// This is a tiny placeholder; real ESS deserves more care.
export function simpleESS(samples: Vector, maxLag = 1000): number {
  const n = samples.length;
  if (n < 3) return n;
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const var0 = samples.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  if (var0 === 0) return n;

  // naive autocorr sum until it goes negative
  let rhoSum = 0;
  for (let lag = 1; lag < Math.min(maxLag, n - 1); lag++) {
    let c = 0;
    for (let i = 0; i < n - lag; i++) {
      c += (samples[i] - mean) * (samples[i + lag] - mean);
    }
    const acf = c / ((n - 1) * var0);
    if (acf <= 0) break;
    rhoSum += 2 * acf;
  }
  return n / (1 + rhoSum);
}

// BDA-style ESS using truncated autocorrelation sum.
// ESS ≈ n / (1 + 2 * Σ ρ_t), with autocorrelations grouped in pairs
// and truncation at the first non-positive pair (ρ_{2k-1} + ρ_{2k} <= 0).
//
// Based on the classical treatment in Gelman et al., Bayesian Data Analysis (3rd ed.),
export function essBDA(samples: Vector, maxLag = 1000): number {
  const n = samples.length;
  if (n < 3) return n;

  const mean = samples.reduce((a, b) => a + b, 0) / n;
  let var0 = 0;
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean;
    var0 += d * d;
  }
  var0 /= n - 1;
  if (var0 === 0) return n;

  const maxPossibleLag = Math.min(maxLag, n - 1);
  const rhos: number[] = [];

  // estimate autocorrelation ρ_t for t = 1..maxPossibleLag
  for (let lag = 1; lag <= maxPossibleLag; lag++) {
    let c = 0;
    for (let i = 0; i < n - lag; i++) {
      c += (samples[i] - mean) * (samples[i + lag] - mean);
    }
    const acf = c / ((n - 1) * var0);
    rhos.push(acf);
  }

  // group in pairs (ρ1+ρ2), (ρ3+ρ4), ..., truncate at first non-positive pair
  let sumRho = 0;
  for (let k = 0; k < rhos.length; k += 2) {
    const rho1 = rhos[k];
    const rho2 = k + 1 < rhos.length ? rhos[k + 1] : 0;
    const pairSum = rho1 + rho2;
    if (pairSum <= 0) break;
    sumRho += pairSum;
  }

  const ess = n / (1 + 2 * sumRho);
  // numerical guard
  return Math.max(1, Math.min(n, ess));
}

// Classic Gelman-Rubin R-hat convergence diagnostic.
// Based on: Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation
// using multiple sequences. Statistical Science, 7(4), 457-472.
export function rhat(chains: Vector[]): number {
  const m = chains.length; // number of chains
  if (m < 2) {
    throw new Error("rhat requires at least 2 chains");
  }

  const n = chains[0].length; // samples per chain
  if (n < 2) {
    throw new Error("rhat requires at least 2 samples per chain");
  }

  // Check all chains have same length
  for (let i = 1; i < m; i++) {
    if (chains[i].length !== n) {
      throw new Error("All chains must have the same length");
    }
  }

  // Compute chain means
  const chainMeans: number[] = [];
  for (let j = 0; j < m; j++) {
    const mean = chains[j].reduce((a, b) => a + b, 0) / n;
    chainMeans.push(mean);
  }

  // Compute overall mean
  const overallMean = chainMeans.reduce((a, b) => a + b, 0) / m;

  // Compute between-chain variance B
  let B = 0;
  for (let j = 0; j < m; j++) {
    const diff = chainMeans[j] - overallMean;
    B += diff * diff;
  }
  B = (n / (m - 1)) * B;

  // Compute within-chain variance W
  let W = 0;
  for (let j = 0; j < m; j++) {
    let chainVar = 0;
    for (let i = 0; i < n; i++) {
      const diff = chains[j][i] - chainMeans[j];
      chainVar += diff * diff;
    }
    W += chainVar / (n - 1);
  }
  W = W / m;

  // Compute pooled variance estimate
  const varPlus = ((n - 1) / n) * W + B / n;

  // Avoid division by zero
  if (W === 0 && varPlus === 0) {
    // All chains are identical constants - perfect convergence
    return 1.0;
  } else if (W === 0) {
    // Within-chain variance is zero but between-chain variance is not
    // This indicates chains have not mixed at all
    return Infinity;
  }

  // R-hat
  return Math.sqrt(varPlus / W);
}

// Compute R-hat for a specific parameter across multiple chains
// samples: [chain][draw][parameter]
export function rhatParam(samples: Vector[][], paramIndex: number): number {
  const chains = samples.map((chain) => chain.map((draw) => draw[paramIndex]));
  return rhat(chains);
}

// Compute R-hat for all parameters
// Returns an array of R-hat values, one per parameter
export function rhatAll(samples: Vector[][]): number[] {
  if (samples.length === 0 || samples[0].length === 0) {
    return [];
  }

  const nParams = samples[0][0].length;
  const rhats: number[] = [];

  for (let p = 0; p < nParams; p++) {
    rhats.push(rhatParam(samples, p));
  }

  return rhats;
}
