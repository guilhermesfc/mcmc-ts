import { describe, it, expect } from "vitest";
import { simpleESS, essBDA, summarizeChains } from "../src/diagnostics";
import { PCG32, Vector } from "../src/core";

describe("simpleESS", () => {
  it("returns n for independent samples", () => {
    // Independent samples should have ESS close to n
    const rng = new PCG32(42n);
    const independent = Array.from({ length: 1000 }, () => rng.normal());
    const ess = simpleESS(independent);
    expect(ess).toBeGreaterThan(500);
  });

  it("returns low ESS for highly correlated samples", () => {
    // Random walk has high autocorrelation
    const correlated: Vector = [0];
    for (let i = 1; i < 1000; i++) {
      correlated.push(correlated[i - 1] + 0.01);
    }
    const ess = simpleESS(correlated);
    expect(ess).toBeLessThan(100);
  });

  it("handles small sample sizes", () => {
    expect(simpleESS([1, 2])).toBe(2);
    expect(simpleESS([1])).toBe(1);
  });

  it("handles constant samples", () => {
    const constant = Array(100).fill(5);
    expect(simpleESS(constant)).toBe(100);
  });

  it("respects maxLag parameter", () => {
    // Create correlated samples with long autocorrelation
    const correlated: Vector = [0];
    for (let i = 1; i < 500; i++) {
      correlated.push(correlated[i - 1] * 0.95 + 0.1);
    }

    // ESS with small maxLag should be higher (less autocorr summed)
    const essSmallLag = simpleESS(correlated, 10);
    // ESS with large maxLag should be lower (more autocorr summed)
    const essLargeLag = simpleESS(correlated, 200);

    expect(essSmallLag).toBeGreaterThan(essLargeLag);
  });
});

describe("essBDA", () => {
  it("returns n for independent samples", () => {
    const rng = new PCG32(42n);
    const independent = Array.from({ length: 1000 }, () => rng.normal());
    const ess = essBDA(independent);
    expect(ess).toBeGreaterThan(500);
  });

  it("returns low ESS for highly correlated samples", () => {
    const correlated: Vector = [0];
    for (let i = 1; i < 1000; i++) {
      correlated.push(correlated[i - 1] + 0.01);
    }
    const ess = essBDA(correlated);
    expect(ess).toBeLessThan(100);
  });

  it("handles small sample sizes", () => {
    expect(essBDA([1, 2])).toBe(2);
    expect(essBDA([1])).toBe(1);
  });

  it("returns value between 1 and n", () => {
    const rng = new PCG32(123n);
    const samples = Array.from({ length: 500 }, () => rng.normal());
    const ess = essBDA(samples);
    expect(ess).toBeGreaterThanOrEqual(1);
    expect(ess).toBeLessThanOrEqual(500);
  });

  it("respects maxLag parameter", () => {
    // Create correlated samples with long autocorrelation
    const correlated: Vector = [0];
    for (let i = 1; i < 500; i++) {
      correlated.push(correlated[i - 1] * 0.95 + 0.1);
    }

    // ESS with small maxLag should be higher (less autocorr summed)
    const essSmallLag = essBDA(correlated, 10);
    // ESS with large maxLag should be lower (more autocorr summed)
    const essLargeLag = essBDA(correlated, 200);

    expect(essSmallLag).toBeGreaterThan(essLargeLag);
  });
});

describe("summarizeChains", () => {
  it("computes correct mean and sd for multiple chains", () => {
    // 2 chains, 4 draws each, 1 parameter
    // Chain 1: [1, 2, 3, 4] -> mean=2.5
    // Chain 2: [5, 6, 7, 8] -> mean=6.5
    // Combined: [1,2,3,4,5,6,7,8] -> mean=4.5, sd≈2.449
    const samples: Vector[][] = [
      [[1], [2], [3], [4]],
      [[5], [6], [7], [8]],
    ];

    const summary = summarizeChains(samples);

    expect(summary.mean[0]).toBeCloseTo(4.5);
    expect(summary.sd[0]).toBeCloseTo(2.449, 2);
    expect(summary.ess.length).toBe(1);
    expect(summary.rhat.length).toBe(1);
    expect(summary.rhat[0]).toBeGreaterThan(1); // chains not converged
  });

  it("returns NaN for rhat with single chain", () => {
    const samples: Vector[][] = [[[1], [2], [3], [4]]];

    const summary = summarizeChains(samples);

    expect(summary.mean[0]).toBeCloseTo(2.5);
    expect(summary.rhat[0]).toBeNaN();
  });

  it("handles multiple parameters", () => {
    const rng = new PCG32(42n);
    // 2 chains, 100 draws, 3 parameters
    const samples: Vector[][] = [];
    for (let c = 0; c < 2; c++) {
      const chain: Vector[] = [];
      for (let d = 0; d < 100; d++) {
        chain.push([rng.normal(), rng.normal() + 5, rng.normal() * 2]);
      }
      samples.push(chain);
    }

    const summary = summarizeChains(samples);

    expect(summary.mean.length).toBe(3);
    expect(summary.sd.length).toBe(3);
    expect(summary.ess.length).toBe(3);
    expect(summary.rhat.length).toBe(3);

    // Second parameter should have mean around 5
    expect(summary.mean[1]).toBeGreaterThan(4);
    expect(summary.mean[1]).toBeLessThan(6);

    // Third parameter should have higher SD (scaled by 2)
    expect(summary.sd[2]).toBeGreaterThan(summary.sd[0]);
  });

  it("works with converged chains", () => {
    const rng = new PCG32(123n);
    // 4 chains sampling from same distribution
    const samples: Vector[][] = [];
    for (let c = 0; c < 4; c++) {
      const chain: Vector[] = [];
      for (let d = 0; d < 500; d++) {
        chain.push([rng.normal()]);
      }
      samples.push(chain);
    }

    const summary = summarizeChains(samples);

    // Mean should be close to 0
    expect(Math.abs(summary.mean[0])).toBeLessThan(0.2);
    // SD should be close to 1
    expect(summary.sd[0]).toBeGreaterThan(0.8);
    expect(summary.sd[0]).toBeLessThan(1.2);
    // R-hat should be close to 1 (converged)
    expect(summary.rhat[0]).toBeGreaterThan(0.99);
    expect(summary.rhat[0]).toBeLessThan(1.1);
    // ESS should be reasonable
    expect(summary.ess[0]).toBeGreaterThan(100);
  });
});
