import { describe, it, expect } from "vitest";
import {
  metropolisHastings,
  rhatAll,
  positiveTransform,
  unitIntervalTransform,
} from "../src";
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

describe("metropolisHastings with transforms", () => {
  it("samples from half-normal using positive transform", () => {
    // Half-normal: p(x) ∝ exp(-0.5 * x²) for x > 0
    const halfNormalLogDensity = (x: Vector) => {
      if (x[0] <= 0) return -Infinity;
      return -0.5 * x[0] * x[0];
    };

    const result = metropolisHastings(halfNormalLogDensity, 1, {
      transforms: [positiveTransform()],
      iterations: 5000,
      burnIn: 500,
      thin: 2,
      start: [1.0], // constrained space (positive)
      stepSize: 0.7,
    });

    const chain = result.samples[0];
    const xs = chain.map((s) => s[0]);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

    // Theoretical mean of half-normal(0, 1) is sqrt(2/π) ≈ 0.798
    const theoreticalMean = Math.sqrt(2 / Math.PI);
    expect(Math.abs(mean - theoreticalMean)).toBeLessThan(0.1);

    // All samples should be positive (constrained space)
    xs.forEach((x) => expect(x).toBeGreaterThan(0));
  });

  it("handles multi-dimensional transforms", () => {
    // Both dimensions constrained
    const logDensity = (x: Vector) => {
      if (x[0] <= 0 || x[1] <= 0 || x[1] >= 1) return -Infinity;
      return -0.5 * x[0] * x[0] - 0.5 * x[1] * x[1];
    };

    const result = metropolisHastings(logDensity, 2, {
      transforms: [positiveTransform(), unitIntervalTransform()],
      iterations: 2000,
      burnIn: 200,
      start: [1.0, 0.5], // constrained: x0 > 0, x1 ∈ (0, 1)
    });

    const chain = result.samples[0];

    // Check all samples are in constrained space
    chain.forEach((sample) => {
      expect(sample[0]).toBeGreaterThan(0); // positive
      expect(sample[1]).toBeGreaterThan(0); // in (0, 1)
      expect(sample[1]).toBeLessThan(1);
    });
  });

  it("transforms user-provided starting points correctly", () => {
    const halfNormalLogDensity = (x: Vector) => {
      if (x[0] <= 0) return -Infinity;
      return -0.5 * x[0] * x[0];
    };

    const result = metropolisHastings(halfNormalLogDensity, 1, {
      transforms: [positiveTransform()],
      iterations: 500,
      start: [2.0], // constrained space
    });

    // Should run without error and produce positive samples
    expect(result.samples[0].length).toBeGreaterThan(0);
    result.samples[0].forEach((s) => expect(s[0]).toBeGreaterThan(0));
  });

  it("works with multiple chains and transforms", () => {
    const halfNormalLogDensity = (x: Vector) => {
      if (x[0] <= 0) return -Infinity;
      return -0.5 * x[0] * x[0];
    };

    const result = metropolisHastings(halfNormalLogDensity, 1, {
      transforms: [positiveTransform()],
      chains: 4,
      iterations: 1000,
      burnIn: 100,
    });

    expect(result.samples.length).toBe(4);

    // All samples in all chains should be positive
    result.samples.forEach((chain) => {
      chain.forEach((sample) => {
        expect(sample[0]).toBeGreaterThan(0);
      });
    });
  });

  it("throws error if transforms length doesn't match dim", () => {
    const logDensity = (x: Vector) => -0.5 * x[0] * x[0];

    expect(() => {
      metropolisHastings(logDensity, 2, {
        transforms: [positiveTransform()], // only 1 transform for 2 dimensions
        iterations: 100,
      });
    }).toThrow("transforms.length (1) must equal dim (2)");
  });

  it("works with custom chainStarts in constrained space", () => {
    const halfNormalLogDensity = (x: Vector) => {
      if (x[0] <= 0) return -Infinity;
      return -0.5 * x[0] * x[0];
    };

    const constrainedStarts: Vector[] = [[0.5], [1.0], [2.0]];

    const result = metropolisHastings(halfNormalLogDensity, 1, {
      transforms: [positiveTransform()],
      chains: 3,
      chainStarts: constrainedStarts, // in constrained space
      iterations: 500,
    });

    expect(result.samples.length).toBe(3);

    // All samples should be positive
    result.samples.forEach((chain) => {
      chain.forEach((sample) => {
        expect(sample[0]).toBeGreaterThan(0);
      });
    });
  });
});
