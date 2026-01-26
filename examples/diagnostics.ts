import { defineModel, Vector } from "../src/index.js";

// Standard normal log density
const logDensity = (x: Vector) => -0.5 * x[0] * x[0];

console.log("=== Example 1: Standard Normal (Good Convergence) ===\n");

const model = defineModel({ logDensity, dim: 1 });

const result = model.sample({
  chains: 4,
  iterations: 100_000,
  warmup: 500,
  thin: 5,
  stepSize: 0.7,
});

const summary = result.summary();

console.log("Chains:", result.draws.length);
console.log("Samples per chain:", result.draws[0].length);
console.log(
  "\nAcceptance rates:",
  result.acceptanceRates.map((r) => r.toFixed(3)).join(", "),
);
console.log("\n=== Summary ===");
console.log("Mean:", summary.mean[0].toFixed(4), "(expected: 0.0000)");
console.log("SD:  ", summary.sd[0].toFixed(4), "(expected: 1.0000)");
console.log("ESS: ", Math.round(summary.ess[0]));
console.log(
  "R-hat:",
  summary.rhat[0].toFixed(4),
  summary.rhat[0] < 1.01 ? "" : "",
);

// Example 2: Banana distribution (difficult to sample)
console.log("\n\n=== Example 2: Banana Distribution (Non-Convergence) ===\n");

const bananaLogDensity = (x: Vector) => {
  const beta = 0.03;
  const logPx = (-0.5 * x[0] * x[0]) / 100;
  const logPy = -0.5 * Math.pow(x[1] - beta * (x[0] * x[0] - 100), 2);
  return logPx + logPy;
};

const bananaModel = defineModel({ logDensity: bananaLogDensity, dim: 2 });

const badResult = bananaModel.sample({
  chains: 4,
  iterations: 10_000,
  warmup: 500,
  stepSize: 1.0,
});

const badSummary = badResult.summary();

console.log("Chains:", badResult.draws.length);
console.log("Samples per chain:", badResult.draws[0].length);
console.log("\n=== Summary ===");
console.log("       Mean      SD     ESS    R-hat");
console.log(
  `x:   ${badSummary.mean[0].toFixed(4).padStart(7)}  ${badSummary.sd[0].toFixed(4).padStart(6)}  ${Math.round(badSummary.ess[0]).toString().padStart(5)}   ${badSummary.rhat[0].toFixed(4)}`,
);
console.log(
  `y:   ${badSummary.mean[1].toFixed(4).padStart(7)}  ${badSummary.sd[1].toFixed(4).padStart(6)}  ${Math.round(badSummary.ess[1]).toString().padStart(5)}   ${badSummary.rhat[1].toFixed(4)}`,
);

const maxRhat = Math.max(...badSummary.rhat);
if (maxRhat > 1.05) {
  console.log("\n Chains have NOT converged (R-hat > 1.05)");
  console.log(
    "   The curved geometry requires more advanced methods (HMC, NUTS)",
  );
}
