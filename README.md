# 🧮 mcmc-ts

A lightweight **Markov Chain Monte Carlo** library in **TypeScript**, featuring a clean implementation of the **Metropolis–Hastings** algorithm.

---

## 🚀 What this is

`mcmc-ts` lets you sample from arbitrary probability distributions using Markov Chain Monte Carlo (MCMC) methods — starting with the **Metropolis–Hastings random walk sampler**.

It’s written entirely in **TypeScript**, with zero dependencies, making it portable for both Node.js and browser environments.

---

## 📦 Installation

After publishing to npm:

```bash
npm install mcmc-ts
# or
yarn add mcmc-ts
# or
pnpm add mcmc-ts
```

---

## 🧑‍💻 Quick Example

Here’s how to sample from a standard normal distribution using Metropolis–Hastings:

```typescript
import { metropolisHastings, zeros, column, simpleESS } from "mcmc-ts";

// log-density of N(0,1)
const logDensity = (x: number[]) => -0.5 * x[0] * x[0];

const result = metropolisHastings(logDensity, 1, {
  iterations: 10_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
  start: [5],
});

const xs = column(result.chain, 0);
const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

console.log("Samples:", xs.length);
console.log("Acceptance rate:", result.acceptanceRate.toFixed(3));
console.log("Sample mean:", mean.toFixed(3));
console.log("ESS:", Math.round(simpleESS(xs)));
```
