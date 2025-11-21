import { Vector } from "./core.js";

// Univariate ESS (very rough): batch means or simple autocorr truncation.
// This is a tiny placeholder; real ESS deserves more care.
export function simpleESS(samples: number[], maxLag = 1000): number {
  const n = samples.length;
  if (n < 3) return n;
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const var0 = samples.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  if (var0 === 0) return n;

  // naive autocorr sum until it goes negative
  let rhoSum = 0;
  for (let lag = 1; lag < Math.min(1000, n - 1); lag++) {
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
export function essBDA(samples: number[], maxLag = 1000): number {
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
