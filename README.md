# 🧮 mcmc-ts

[![npm version](https://img.shields.io/npm/v/mcmc-ts)](https://www.npmjs.com/package/mcmc-ts)
[![license](https://img.shields.io/npm/l/mcmc-ts)](https://github.com/guilhermesfc/mcmc-ts/blob/main/LICENSE)
[![npm downloads](https://img.shields.io/npm/dm/mcmc-ts)](https://www.npmjs.com/package/mcmc-ts)

A lightweight **Markov Chain Monte Carlo** library in **TypeScript**, featuring a clean implementation of the **Metropolis–Hastings** algorithm.

---

## 🚀 What this is

`mcmc-ts` lets you sample from arbitrary probability distributions using Markov Chain Monte Carlo (MCMC) methods — starting with the **Metropolis–Hastings random walk sampler**.

It's written entirely in **TypeScript**, with zero dependencies, making it portable for both Node.js and browser environments.

---

## 📦 Installation

```bash
npm install mcmc-ts
# or
yarn add mcmc-ts
# or
pnpm add mcmc-ts
```

---

## 🧑‍💻 Quick Example

Here's how to sample from a standard normal distribution using Metropolis–Hastings:

```typescript
import { metropolisHastings, simpleESS } from "mcmc-ts";

// log-density of N(0,1)
const logDensity = (x: number[]) => -0.5 * x[0] * x[0];

const result = metropolisHastings(logDensity, 1, {
  iterations: 10_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
  start: [5],
});

// Extract samples from first (and only) chain
const chain = result.samples[0];
const xs = chain.map((row) => row[0]);
const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

console.log("Samples:", xs.length);
console.log("Acceptance rate:", result.acceptanceRates[0].toFixed(3));
console.log("Sample mean:", mean.toFixed(3));
console.log("ESS:", Math.round(simpleESS(xs)));
```

**Output:**

```
Samples: 1901
Acceptance rate: 0.789
Sample mean: 0.003
ESS: 710
```

<img src="assets/example-output.png" alt="MCMC Visualization" width="626">

---

## 🔗 Multiple Chains & Convergence

Run multiple chains to assess convergence:

```typescript
import { metropolisHastings, rhatAll } from "mcmc-ts";

const result = metropolisHastings(logDensity, 1, {
  chains: 4, // Run 4 chains
  iterations: 10_000,
  burnIn: 500,
  stepSize: 0.7,
});

// Check convergence with R-hat
const rhats = rhatAll(result.samples);
console.log("R-hat:", rhats[0]); // Should be < 1.01 for good convergence
```

---

## 🎯 Constrained Sampling

Use transforms for constrained distributions:

```typescript
import { metropolisHastings, positiveTransform } from "mcmc-ts";

const result = metropolisHastings(halfNormalLogDensity, 1, {
  transforms: [positiveTransform()], // Handles x > 0 constraint
  start: [1.0],
  iterations: 10_000,
});
```

Available: `positiveTransform()`, `unitIntervalTransform()`, `simplexTransform()`

---

## 📚 API

**Samplers**

- `metropolisHastings(logDensity, dim, options)` - Metropolis-Hastings sampler
  - Set `options.chains` to run multiple chains (default: 1)
  - Set `options.transforms` for constrained sampling (optional)
  - Set `options.seed` for reproducible results (default: 0n)
  - Returns `samples` as `[chain][draw]` array (always 2D, even for single chain)

**Diagnostics**

- `simpleESS(samples)`, `essBDA(samples)` - Effective sample size
- `rhat(chains)`, `rhatAll(samples)` - Gelman-Rubin convergence diagnostic

**Transforms**

- `positiveTransform()`, `unitIntervalTransform()`, `simplexTransform()` - Constraint handlers
