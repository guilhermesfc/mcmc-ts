import { describe, it, expect } from "vitest";
import { metropolisHastings, rhatAll } from "../src";
import { Vector } from "../src/core";

describe("metropolisHastings", () => {
  it("samples from standard normal distribution (single chain)", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];
    const result = metropolisHastings(logp, 1, {
      iterations: 5000,
      stepSize: 1.0,
      burnIn: 500,
      thin: 2,
      start: [3],
    });

    // Access first (and only) chain
    const chain = result.samples[0];
    const acceptanceRate = result.acceptanceRates[0];

    const xs = chain.map((s) => s[0]);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(acceptanceRate).toBeGreaterThan(0.1);
    expect(acceptanceRate).toBeLessThan(0.9);

    // Verify structure: always has chain dimension
    expect(result.samples.length).toBe(1); // 1 chain
    expect(result.acceptanceRates.length).toBe(1);
  });

  it("runs multiple chains when specified", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];
    const result = metropolisHastings(logp, 1, {
      chains: 4,
      iterations: 1000,
      burnIn: 100,
      stepSize: 0.7,
    });

    expect(result.samples.length).toBe(4); // 4 chains
    expect(result.acceptanceRates.length).toBe(4);
    expect(result.rawTraces.length).toBe(4);

    // Each chain should have samples
    for (let i = 0; i < 4; i++) {
      expect(result.samples[i].length).toBeGreaterThan(0);
      expect(result.acceptanceRates[i]).toBeGreaterThan(0);
      expect(result.acceptanceRates[i]).toBeLessThanOrEqual(1);
    }

    // Check convergence with R-hat
    const rhats = rhatAll(result.samples);
    expect(rhats[0]).toBeGreaterThan(0);
    expect(rhats[0]).toBeLessThan(2);
  });

  it("accepts custom chain starting points", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    const starts: Vector[] = [[-2], [0], [2]];

    const result = metropolisHastings(logp, 1, {
      chains: 3,
      chainStarts: starts,
      iterations: 500,
      stepSize: 0.5,
    });

    expect(result.samples.length).toBe(3);
  });

  it("throws error if chainStarts length doesn't match chains", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    expect(() => {
      metropolisHastings(logp, 1, {
        chains: 3,
        chainStarts: [[0], [1]], // only 2 starts for 3 chains
        iterations: 100,
      });
    }).toThrow();
  });
});
