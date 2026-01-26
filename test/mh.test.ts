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
      transforms: [{ startIndex: 0, transform: positiveTransform() }],
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
      transforms: [
        { startIndex: 0, transform: positiveTransform() },
        { startIndex: 1, transform: unitIntervalTransform() },
      ],
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
      transforms: [{ startIndex: 0, transform: positiveTransform() }],
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
      transforms: [{ startIndex: 0, transform: positiveTransform() }],
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

  it("throws error if transforms exceed dim", () => {
    const logDensity = (x: Vector) => -0.5 * x[0] * x[0];

    expect(() => {
      metropolisHastings(logDensity, 1, {
        transforms: [{ startIndex: 0, transform: positiveTransform(2) }], // 2-dim transform for 1-dim problem
        iterations: 100,
      });
    }).toThrow(/exceed/);
  });

  it("works with custom chainStarts in constrained space", () => {
    const halfNormalLogDensity = (x: Vector) => {
      if (x[0] <= 0) return -Infinity;
      return -0.5 * x[0] * x[0];
    };

    const constrainedStarts: Vector[] = [[0.5], [1.0], [2.0]];

    const result = metropolisHastings(halfNormalLogDensity, 1, {
      transforms: [{ startIndex: 0, transform: positiveTransform() }],
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

describe("metropolisHastings with adaptive step size", () => {
  it("adapts step size to achieve target acceptance rate", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    // Start with a reasonable step size that needs some adjustment
    const result = metropolisHastings(logp, 1, {
      iterations: 10000,
      stepSize: 0.5,
      targetAcceptance: 0.23,
      adaptSteps: 5000,
      adaptWindow: 50,
      burnIn: 5000,
    });

    // After adaptation, acceptance rate should be closer to target
    const acceptanceRate = result.acceptanceRates[0];
    expect(acceptanceRate).toBeGreaterThan(0.1);
    expect(acceptanceRate).toBeLessThan(0.5);
  });

  it("does not adapt when targetAcceptance is not provided", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    // Without targetAcceptance, behavior should be unchanged
    const result = metropolisHastings(logp, 1, {
      iterations: 2000,
      stepSize: 0.01, // Very small step size
      burnIn: 100,
    });

    // With such a small step size and no adaptation, acceptance should be very high
    expect(result.acceptanceRates[0]).toBeGreaterThan(0.9);
  });

  it("adapts independently per chain", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    const result = metropolisHastings(logp, 1, {
      chains: 4,
      iterations: 10000,
      stepSize: 0.5,
      targetAcceptance: 0.23,
      adaptSteps: 5000,
      burnIn: 5000,
    });

    // All chains should have adapted to reasonable acceptance rates
    result.acceptanceRates.forEach((rate) => {
      expect(rate).toBeGreaterThan(0.1);
      expect(rate).toBeLessThan(0.5);
    });
  });

  it("works with transforms and adaptive step size", () => {
    const halfNormalLogDensity = (x: Vector) => {
      if (x[0] <= 0) return -Infinity;
      return -0.5 * x[0] * x[0];
    };

    const result = metropolisHastings(halfNormalLogDensity, 1, {
      transforms: [{ startIndex: 0, transform: positiveTransform() }],
      iterations: 10000,
      burnIn: 5000,
      stepSize: 1.0, // Closer to optimal to avoid numerical issues
      targetAcceptance: 0.23,
      adaptSteps: 5000,
      start: [1.0],
    });

    const chain = result.samples[0];
    const xs = chain.map((s) => s[0]);

    // All samples should be positive
    xs.forEach((x) => expect(x).toBeGreaterThan(0));

    // Acceptance rate should be reasonable after adaptation
    expect(result.acceptanceRates[0]).toBeGreaterThan(0.1);
    expect(result.acceptanceRates[0]).toBeLessThan(0.5);
  });

  it("stops adapting after adaptSteps", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    // Run two samplers with different adaptSteps but same total iterations
    const shortAdapt = metropolisHastings(logp, 1, {
      iterations: 10000,
      stepSize: 0.5,
      targetAcceptance: 0.23,
      adaptSteps: 1000, // Short adaptation
      seed: 42n,
    });

    const longAdapt = metropolisHastings(logp, 1, {
      iterations: 10000,
      stepSize: 0.5,
      targetAcceptance: 0.23,
      adaptSteps: 8000, // Long adaptation
      seed: 42n,
    });

    // Both should run without error
    expect(shortAdapt.samples[0].length).toBeGreaterThan(0);
    expect(longAdapt.samples[0].length).toBeGreaterThan(0);

    // Both should have reasonable acceptance rates
    expect(shortAdapt.acceptanceRates[0]).toBeGreaterThan(0.1);
    expect(longAdapt.acceptanceRates[0]).toBeGreaterThan(0.1);
  });
});

describe("metropolisHastings with per-dimension step sizes", () => {
  it("accepts an array of step sizes", () => {
    // 2D Gaussian with different scales: x ~ N(0,1), y ~ N(0,100)
    const logp = (x: Vector) =>
      -0.5 * x[0] * x[0] - (0.5 * (x[1] * x[1])) / 100;

    const result = metropolisHastings(logp, 2, {
      iterations: 5000,
      stepSize: [0.5, 5.0], // Different step sizes per dimension
      burnIn: 500,
    });

    const chain = result.samples[0];
    const xs = chain.map((s) => s[0]);
    const ys = chain.map((s) => s[1]);

    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;

    // Both means should be close to 0
    expect(Math.abs(meanX)).toBeLessThan(0.2);
    expect(Math.abs(meanY)).toBeLessThan(2);

    // Acceptance rate should be reasonable
    expect(result.acceptanceRates[0]).toBeGreaterThan(0.1);
    expect(result.acceptanceRates[0]).toBeLessThan(0.9);
  });

  it("throws error if stepSize array length doesn't match unconstrained dim", () => {
    const logp = (x: Vector) => -0.5 * x[0] * x[0];

    expect(() => {
      metropolisHastings(logp, 2, {
        stepSize: [0.5], // Only 1 step size for 2 dimensions
        iterations: 100,
      });
    }).toThrow("stepSize array length (1) must equal unconstrained dim (2)");
  });

  it("works with adaptation and per-dimension step sizes", () => {
    // 2D Gaussian with different scales
    const logp = (x: Vector) =>
      -0.5 * x[0] * x[0] - (0.5 * (x[1] * x[1])) / 100;

    const result = metropolisHastings(logp, 2, {
      iterations: 10000,
      stepSize: [0.5, 5.0],
      targetAcceptance: 0.23,
      adaptSteps: 5000,
      burnIn: 5000,
    });

    // Should have reasonable acceptance after adaptation
    expect(result.acceptanceRates[0]).toBeGreaterThan(0.1);
    expect(result.acceptanceRates[0]).toBeLessThan(0.5);
  });
});
