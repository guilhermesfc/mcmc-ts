import { metropolisHastings, essBDA, rhatAll, Vector } from "../src/index.js";

const dim = 1;

// Standard normal log density (up to a constant)
const logDensity = (x: Vector) => {
  const v = x[0];
  return -0.5 * v * v;
};

console.log("=== Example 1: Standard Normal (Good Convergence) ===");
console.log("This is a simple 1D Gaussian distribution that should");
console.log("converge easily with random-walk Metropolis-Hastings.\n");

// Run 4 chains for convergence diagnostics
const res = metropolisHastings(logDensity, dim, {
  chains: 4,
  iterations: 100_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
});

console.log("Chains:", res.samples.length);
console.log("Samples per chain:", res.samples[0].length);

// Per-chain acceptance rates
console.log("\n=== Acceptance Rates ===");
res.acceptanceRates.forEach((rate, i) => {
  console.log(`Chain ${i + 1}: ${rate.toFixed(3)}`);
});

// Per-chain means
console.log("\n=== Chain Means (expected: ~0.0000) ===");
const chainMeans = res.samples.map((chain) => {
  const xs = chain.map((v) => v[0]);
  return xs.reduce((a, b) => a + b, 0) / xs.length;
});
chainMeans.forEach((mean, i) => {
  console.log(`Chain ${i + 1}: ${mean.toFixed(4)}`);
});

// Convergence diagnostic
console.log("\n=== Convergence Diagnostics ===");
const rhats = rhatAll(res.samples);
console.log(`R-hat: ${rhats[0].toFixed(4)}`);
if (rhats[0] < 1.01) {
  console.log("✓ Excellent convergence (R-hat < 1.01)");
} else if (rhats[0] < 1.05) {
  console.log("✓ Good convergence (R-hat < 1.05)");
} else {
  console.log("⚠ Chains may not have converged (R-hat > 1.05)");
}

// Per-chain ESS
console.log("\n=== Effective Sample Size (per chain) ===");
res.samples.forEach((chain, i) => {
  const xs = chain.map((v) => v[0]);
  const ess = essBDA(xs);
  console.log(
    `Chain ${i + 1}: ${Math.round(ess)} (${(ess / xs.length).toFixed(2)})`,
  );
});

// Combined statistics (all chains)
const allSamples = res.samples.flat();
const xs = allSamples.map((row) => row[0]);
const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
const variance = xs.reduce((s, v) => s + (v - mean) ** 2, 0) / (xs.length - 1);
const std = Math.sqrt(variance);

console.log("\n=== Combined Statistics (all chains) ===");
console.log("Total samples:", xs.length);
console.log("Mean:", mean.toFixed(4), "(expected: 0.0000)");
console.log("Std:", std.toFixed(4), "(expected: 1.0000)");
console.log("ESS:", Math.round(essBDA(xs)));

// Example 2: Banana distribution (difficult to sample)
console.log("\n\n=== Example 2: Banana Distribution (Non-Convergence) ===");
console.log("This distribution has a curved, banana-like shape that is");
console.log("difficult for random-walk Metropolis-Hastings to explore.\n");

// Banana-shaped distribution: x ~ N(0, 100), y ~ N(β(x²-100), 1) with β=0.03
const bananaLogDensity = (x: Vector) => {
  const beta = 0.03;
  const x0 = x[0];
  const x1 = x[1];
  const logPx = (-0.5 * (x0 * x0)) / 100; // x ~ N(0, 100)
  const logPy = -0.5 * Math.pow(x1 - beta * (x0 * x0 - 100), 2); // y ~ N(β(x²-100), 1)
  return logPx + logPy;
};

const badRes = metropolisHastings(bananaLogDensity, 2, {
  chains: 4,
  iterations: 10_000,
  stepSize: 1.0,
  burnIn: 500,
  thin: 1,
});

console.log("Chains:", badRes.samples.length);
console.log("Samples per chain:", badRes.samples[0].length);

// Per-chain acceptance rates
console.log("\n=== Acceptance Rates ===");
badRes.acceptanceRates.forEach((rate, i) => {
  console.log(`Chain ${i + 1}: ${rate.toFixed(3)}`);
});

// Per-chain means for x (dimension 0)
console.log("\n=== Chain Means (x dimension, expected: ~0.0000) ===");
const badChainMeansX = badRes.samples.map((chain) => {
  const xs = chain.map((v) => v[0]);
  return xs.reduce((a, b) => a + b, 0) / xs.length;
});
badChainMeansX.forEach((mean, i) => {
  console.log(`Chain ${i + 1}: ${mean.toFixed(4)}`);
});

// Per-chain means for y (dimension 1)
console.log("\n=== Chain Means (y dimension, expected: ~0.0000) ===");
const badChainMeansY = badRes.samples.map((chain) => {
  const ys = chain.map((v) => v[1]);
  return ys.reduce((a, b) => a + b, 0) / ys.length;
});
badChainMeansY.forEach((mean, i) => {
  console.log(`Chain ${i + 1}: ${mean.toFixed(4)}`);
});

// R-hat
console.log("\n=== Convergence Diagnostics ===");
const badRhats = rhatAll(badRes.samples);
console.log(`R-hat (x): ${badRhats[0].toFixed(4)}`);
console.log(`R-hat (y): ${badRhats[1].toFixed(4)}`);

const maxRhat = Math.max(...badRhats);
if (maxRhat < 1.01) {
  console.log("✓ Excellent convergence (R-hat < 1.01)");
} else if (maxRhat < 1.05) {
  console.log("✓ Good convergence (R-hat < 1.05)");
} else {
  console.log("⚠ Chains have NOT converged (R-hat > 1.05)");
  console.log(
    "   → The curved geometry makes this distribution hard to sample",
  );
  console.log("   → Would need more advanced methods (e.g., HMC, NUTS)");
}

// ESS will be low
console.log("\n=== Effective Sample Size (per chain, x dimension) ===");
badRes.samples.forEach((chain, i) => {
  const xs = chain.map((v) => v[0]);
  const ess = essBDA(xs);
  console.log(
    `Chain ${i + 1}: ${Math.round(ess)} (${(ess / xs.length).toFixed(2)})`,
  );
});

// Combined statistics (all chains)
const badAllSamples = badRes.samples.flat();
const badXs = badAllSamples.map((row) => row[0]);
const badYs = badAllSamples.map((row) => row[1]);

const badMeanX = badXs.reduce((a, b) => a + b, 0) / badXs.length;
const badMeanY = badYs.reduce((a, b) => a + b, 0) / badYs.length;

const badVarianceX =
  badXs.reduce((s, v) => s + (v - badMeanX) ** 2, 0) / (badXs.length - 1);
const badVarianceY =
  badYs.reduce((s, v) => s + (v - badMeanY) ** 2, 0) / (badYs.length - 1);

const badStdX = Math.sqrt(badVarianceX);
const badStdY = Math.sqrt(badVarianceY);

console.log("\n=== Combined Statistics (all chains) ===");
console.log("Total samples:", badAllSamples.length);
console.log("Mean (x):", badMeanX.toFixed(4), "(expected: 0.0000)");
console.log("Mean (y):", badMeanY.toFixed(4), "(expected: 0.0000)");
console.log("Std (x):", badStdX.toFixed(4), "(expected: 10.0000)");
console.log("Std (y):", badStdY.toFixed(4), "(expected: ~4.36)");
