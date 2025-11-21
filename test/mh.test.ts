import { describe, it, expect } from "vitest";
import { metropolisHastings } from "../src";

describe("metropolisHastings", () => {
  it("samples from standard normal distribution", () => {
    const logp = (x: number[]) => -0.5 * x[0] * x[0];
    const { chain, acceptanceRate } = metropolisHastings(logp, 1, {
      iterations: 5000,
      stepSize: 1.0,
      burnIn: 500,
      thin: 2,
      start: [3],
    });

    const xs = chain.map((s) => s[0]);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(acceptanceRate).toBeGreaterThan(0.1);
    expect(acceptanceRate).toBeLessThan(0.9);
  });
});
