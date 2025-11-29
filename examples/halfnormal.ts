import {
  metropolisHastings,
  rhatAll,
  simpleESS,
  positiveTransform,
} from "../src/index.js";

// Half-normal log-density (only positive values)
// p(x) ∝ exp(-0.5 * x²) for x > 0
const halfNormalLogDensity = (x: number[]) => {
  if (x[0] <= 0) return -Infinity; // reject negative values
  return -0.5 * x[0] * x[0];
};

console.log("=== Half-Normal Distribution with Positive Transform ===\n");
console.log("The half-normal is restricted to positive values (x > 0).");
console.log(
  "We use positiveTransform() to automatically handle constraints.\n",
);

// Run 4 chains - transforms are handled automatically!
const result = metropolisHastings(halfNormalLogDensity, 1, {
  transforms: [positiveTransform()], // Specify transform in options
  chains: 4,
  iterations: 10_000,
  burnIn: 500,
  thin: 5,
  stepSize: 0.7,
  start: [1.0], // In constrained space (positive value)
});

console.log(`Chains: ${result.samples.length}`);
console.log(`Samples per chain: ${result.samples[0].length}`);
console.log();

// Show acceptance rates
console.log("Acceptance rates:");
result.acceptanceRates.forEach((rate, i) => {
  console.log(`  Chain ${i + 1}: ${rate.toFixed(3)}`);
});
console.log();

// Samples are already in constrained space (positive values) - no manual transform needed!
// Compute means for each chain
console.log("Chain means (expected: ~0.798):");
const chainMeans = result.samples.map((chain) => {
  const xs = chain.map((v) => v[0]);
  return xs.reduce((a, b) => a + b, 0) / xs.length;
});
chainMeans.forEach((mean, i) => {
  console.log(`  Chain ${i + 1}: ${mean.toFixed(3)}`);
});
console.log();

// Compute ESS for each chain
console.log("ESS per chain:");
result.samples.forEach((chain, i) => {
  const xs = chain.map((v) => v[0]);
  const ess = simpleESS(xs);
  console.log(`  Chain ${i + 1}: ${Math.round(ess)}`);
});
console.log();

// Compute R-hat to assess convergence
const rhats = rhatAll(result.samples);
console.log("R-hat diagnostics:");
console.log(`  Parameter 1: ${rhats[0].toFixed(4)}`);
if (rhats[0] < 1.01) {
  console.log("  ✓ Excellent convergence (R-hat < 1.01)");
} else if (rhats[0] < 1.05) {
  console.log("  ✓ Good convergence (R-hat < 1.05)");
} else {
  console.log("  ⚠ Chains may not have converged (R-hat > 1.05)");
}
console.log();

// Combine all chains for final inference
const allSamples = result.samples.flat();
const xs = allSamples.map((v) => v[0]);
const overallMean = xs.reduce((a, b) => a + b, 0) / xs.length;
const variance =
  xs.reduce((s, v) => s + (v - overallMean) ** 2, 0) / (xs.length - 1);
const std = Math.sqrt(variance);

// Theoretical values for half-normal(0, 1)
const theoreticalMean = Math.sqrt(2 / Math.PI); // ≈ 0.798
const theoreticalStd = Math.sqrt(1 - 2 / Math.PI); // ≈ 0.602

console.log("Combined results:");
console.log(`  Total samples: ${xs.length}`);
console.log(
  `  Mean: ${overallMean.toFixed(3)} (expected: ${theoreticalMean.toFixed(3)})`,
);
console.log(
  `  Std: ${std.toFixed(3)} (expected: ${theoreticalStd.toFixed(3)})`,
);
console.log();

console.log(
  "Note: All samples are automatically in constrained space (positive):",
);
console.log(`  Min value: ${Math.min(...xs).toFixed(4)}`);
console.log(`  Max value: ${Math.max(...xs).toFixed(4)}`);
