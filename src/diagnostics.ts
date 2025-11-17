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
  let rhoSUm = 0;
  for (let lag = 1; lag < Math.min(1000, n - 1); lag++) {
    let c = 0;
    for (let i = 0; i < n - lag; i++) {
      c += (samples[i] - mean) * (samples[i + lag] - mean);
    }
    const acf = c / ((n - 1) * var0);
    if (acf <= 0) break;
    rhoSUm += 2 * acf;
  }
  return n / (1 + rhoSUm);
}
