import { defineModel } from "../src/index.js";

// Half-normal log-density (only positive values)
// p(x) ∝ exp(-0.5 * x^2) for x > 0
const halfNormalLogDensity = (x: number[]) => -0.5 * x[0] * x[0];

console.log("=== Half-Normal Distribution with Positive Constraint ===\n");
console.log("The half-normal is restricted to positive values (x > 0).");
console.log("We use constraints: { 0: 'positive' } to handle this automatically.\n");

// Define model with positive constraint
const model = defineModel({
  logDensity: halfNormalLogDensity,
  dim: 1,
  constraints: { 0: "positive" },
});

// Run 4 chains
const result = model.sample({
  chains: 4,
  iterations: 10_000,
  warmup: 500,
  thin: 5,
  stepSize: 0.7,
  start: [1.0],
});

console.log(`Chains: ${result.draws.length}`);
console.log(`Samples per chain: ${result.draws[0].length}`);
console.log();

// Show acceptance rates
console.log("Acceptance rates:");
result.acceptanceRates.forEach((rate, i) => {
  console.log(`  Chain ${i + 1}: ${rate.toFixed(3)}`);
});
console.log();

// Compute means for each chain
console.log("Chain means (expected: ~0.798):");
const chainMeans = result.draws.map((chain) => {
  const xs = chain.map((v) => v[0]);
  return xs.reduce((a, b) => a + b, 0) / xs.length;
});
chainMeans.forEach((mean, i) => {
  console.log(`  Chain ${i + 1}: ${mean.toFixed(3)}`);
});
console.log();

// Get summary
const summary = result.summary();
console.log("R-hat diagnostics:");
console.log(`  Parameter 1: ${summary.rhat[0].toFixed(4)}`);
if (summary.rhat[0] < 1.01) {
  console.log("  Excellent convergence (R-hat < 1.01)");
} else if (summary.rhat[0] < 1.05) {
  console.log("  Good convergence (R-hat < 1.05)");
} else {
  console.log("  Chains may not have converged (R-hat > 1.05)");
}
console.log();

// Combine all chains for final inference
const allSamples = result.draws.flat();
const xs = allSamples.map((v) => v[0]);
const overallMean = xs.reduce((a, b) => a + b, 0) / xs.length;
const variance = xs.reduce((s, v) => s + (v - overallMean) ** 2, 0) / (xs.length - 1);
const std = Math.sqrt(variance);

// Theoretical values for half-normal(0, 1)
const theoreticalMean = Math.sqrt(2 / Math.PI); // ~ 0.798
const theoreticalStd = Math.sqrt(1 - 2 / Math.PI); // ~ 0.602

console.log("Combined results:");
console.log(`  Total samples: ${xs.length}`);
console.log(`  Mean: ${overallMean.toFixed(3)} (expected: ${theoreticalMean.toFixed(3)})`);
console.log(`  Std: ${std.toFixed(3)} (expected: ${theoreticalStd.toFixed(3)})`);
console.log();

console.log("Note: All samples are automatically in constrained space (positive):");
console.log(`  Min value: ${Math.min(...xs).toFixed(4)}`);
console.log(`  Max value: ${Math.max(...xs).toFixed(4)}`);
