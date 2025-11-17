import { metropolisHastings, simpleESS } from "../src/index.js";

const dim = 1;

// Standard normal log density (up to a constant)
const logDensity = (x: number[]) => {
  const v = x[0];
  return -0.5 * v * v;
};

const res = metropolisHastings(logDensity, dim, {
  iterations: 100_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
  start: [5],
});

const xs = res.chain.map((row) => row[0]);
const n = xs.length;
const mean = xs.reduce((a, b) => a + b, 0) / n;
const variance = xs.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
const std = Math.sqrt(variance);

console.log("=== MCMC Diagnostics ===");
console.log("Samples kept:", n);
console.log("Acceptance rate:", res.acceptanceRate.toFixed(3));
console.log("\n=== Distribution Statistics ===");
console.log("Sample mean:", mean.toFixed(4), "(expected: 0.0000)");
console.log("Sample std:", std.toFixed(4), "(expected: 1.0000)");
console.log("Sample variance:", variance.toFixed(4), "(expected: 1.0000)");
console.log("\n=== Effective Sample Size ===");
console.log("ESS:", Math.round(simpleESS(xs)));
console.log("ESS / n:", (simpleESS(xs) / n).toFixed(3));

// Check if starting point affects results
console.log("\n=== Chain Behavior ===");
console.log("Starting value:", res.rawTrace[0][0]);
console.log("First 10 samples (post burn-in):", xs.slice(0, 10).map(x => x.toFixed(2)));
console.log("Min value:", Math.min(...xs).toFixed(2));
console.log("Max value:", Math.max(...xs).toFixed(2));
