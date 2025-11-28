import { describe, it, expect } from "vitest";
import { simpleESS, essBDA } from "../src/diagnostics";
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
