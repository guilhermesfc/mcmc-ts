import { metropolisHastings, simpleESS } from "../src/index.js";

const dim = 1;

// Standard normal log density (up to a constant)
const logDensity = (x: number[]) => {
  const v = x[0];
  return -0.5 * v * v; // no need to add -0.5*log(2π)
};

const res = metropolisHastings(logDensity, dim, {
  iterations: 100_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
  start: [5],
});

const xs = res.chain.map((row) => row[0]);
const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

console.log("Samples kept:", xs.length);
console.log("Acceptance rate:", res.acceptanceRate.toFixed(3));
console.log("Sample mean (should be ~0):", mean.toFixed(3));
console.log("ESS (rough):", Math.round(simpleESS(xs)));
