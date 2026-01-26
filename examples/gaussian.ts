import { defineModel } from "../src/index.js";

// Standard normal log-density
const logDensity = (x: number[]) => -0.5 * x[0] * x[0];

console.log("=== Running Multiple Chains ===\n");

// Define model and run 4 chains
const model = defineModel({ logDensity, dim: 1 });

const result = model.sample({
  chains: 4,
  iterations: 10_000,
  warmup: 500,
  thin: 5,
  stepSize: 0.7,
});

console.log(`Chains: ${result.draws.length}`);
console.log(`Samples per chain: ${result.draws[0].length}`);
console.log();

// Show acceptance rates for each chain
console.log("Acceptance rates:");
result.acceptanceRates.forEach((rate, i) => {
  console.log(`  Chain ${i + 1}: ${rate.toFixed(3)}`);
});
console.log();

// Compute means for each chain
console.log("Chain means:");
const chainMeans = result.draws.map((chain) => {
  const xs = chain.map((v) => v[0]);
  return xs.reduce((a, b) => a + b, 0) / xs.length;
});
chainMeans.forEach((mean, i) => {
  console.log(`  Chain ${i + 1}: ${mean.toFixed(3)}`);
});
console.log();

// Get summary with ESS and R-hat
const summary = result.summary();
console.log("Summary:");
console.log(`  Mean: ${summary.mean[0].toFixed(4)}`);
console.log(`  SD: ${summary.sd[0].toFixed(4)}`);
console.log(`  ESS: ${Math.round(summary.ess[0])}`);
console.log(`  R-hat: ${summary.rhat[0].toFixed(4)}`);

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

console.log("Combined results:");
console.log(`  Total samples: ${xs.length}`);
console.log(`  Mean: ${(xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(3)}`);
