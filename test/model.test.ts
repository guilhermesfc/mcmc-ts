import { describe, it, expect } from "vitest";
import {
  defineModel,
  positiveTransform,
  simplexTransform,
  Vector,
} from "../src/index";

describe("defineModel", () => {
  describe("basic functionality", () => {
    it("creates a model with correct dimensions", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      expect(model.constrainedDim).toBe(1);
      expect(model.unconstrainedDim).toBe(1);
    });

    it("samples from a simple normal distribution", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      expect(result.draws.length).toBe(1); // 1 chain
      expect(result.draws[0].length).toBeGreaterThan(0);
      expect(result.acceptanceRates.length).toBe(1);
    });

    it("supports multiple chains", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      const result = model.sample({
        iterations: 500,
        warmup: 50,
        chains: 4,
        seed: 42n,
      });

      expect(result.draws.length).toBe(4);
      expect(result.acceptanceRates.length).toBe(4);
    });
  });

  describe("constraint syntax", () => {
    it("supports string constraint 'positive'", () => {
      const model = defineModel({
        logDensity: (x) => -x[0], // exponential-like
        dim: 1,
        constraints: { 0: "positive" },
      });

      expect(model.constrainedDim).toBe(1);
      expect(model.unconstrainedDim).toBe(1);

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      // All samples should be positive
      for (const draw of result.draws[0]) {
        expect(draw[0]).toBeGreaterThan(0);
      }
    });

    it("supports object constraint { type: 'bounded' }", () => {
      const model = defineModel({
        logDensity: (_x) => 0, // uniform
        dim: 1,
        constraints: { 0: { type: "bounded", lower: 2, upper: 8 } },
      });

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      // All samples should be in (2, 8)
      for (const draw of result.draws[0]) {
        expect(draw[0]).toBeGreaterThan(2);
        expect(draw[0]).toBeLessThan(8);
      }
    });

    it("supports 'unitInterval' constraint", () => {
      const model = defineModel({
        logDensity: (_x) => 0,
        dim: 1,
        constraints: { 0: "unitInterval" },
      });

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      for (const draw of result.draws[0]) {
        expect(draw[0]).toBeGreaterThan(0);
        expect(draw[0]).toBeLessThan(1);
      }
    });

    it("supports simplex constraint with dimension change", () => {
      // Dirichlet(1,1,1) = uniform on simplex
      const model = defineModel({
        logDensity: (_x) => 0,
        dim: 3,
        constraints: { 0: { type: "simplex", k: 3 } },
      });

      // Simplex reduces dimension by 1
      expect(model.constrainedDim).toBe(3);
      expect(model.unconstrainedDim).toBe(2);

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      // All samples should sum to 1 and be positive
      for (const draw of result.draws[0]) {
        const sum = draw.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1, 5);
        for (const xi of draw) {
          expect(xi).toBeGreaterThan(0);
        }
      }
    });

    it("supports mixed constraints", () => {
      const model = defineModel({
        logDensity: (_x) => 0,
        dim: 3,
        constraints: {
          0: "positive",
          1: "unitInterval",
          // 2 is unconstrained by default
        },
      });

      expect(model.constrainedDim).toBe(3);
      expect(model.unconstrainedDim).toBe(3);

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      for (const draw of result.draws[0]) {
        expect(draw[0]).toBeGreaterThan(0); // positive
        expect(draw[1]).toBeGreaterThan(0); // unit interval
        expect(draw[1]).toBeLessThan(1);
        // draw[2] can be anything
      }
    });

    it("throws when both constraints and transforms specified", () => {
      expect(() =>
        defineModel({
          logDensity: (x) => -x[0],
          dim: 1,
          constraints: { 0: "positive" },
          transforms: [{ startIndex: 0, transform: positiveTransform() }],
        }),
      ).toThrow(/both/i);
    });
  });

  describe("explicit transforms", () => {
    it("supports explicit TransformSpec", () => {
      const model = defineModel({
        logDensity: (x) => -x[0],
        dim: 1,
        transforms: [{ startIndex: 0, transform: positiveTransform() }],
      });

      const result = model.sample({
        iterations: 1000,
        warmup: 100,
        seed: 42n,
      });

      for (const draw of result.draws[0]) {
        expect(draw[0]).toBeGreaterThan(0);
      }
    });
  });

  describe("sample options", () => {
    it("separates warmup and post-warmup draws", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      const result = model.sample({
        iterations: 500,
        warmup: 100,
        thin: 1,
        includeWarmup: true,
        seed: 42n,
      });

      // total samples = iterations + warmup + 1 (initial state) = 601
      // warmupCount = floor(warmup / thin) = 100
      // warmupDraws = first 100, draws = remaining 501
      expect(result.warmupDraws[0].length).toBe(100);
      expect(result.draws[0].length).toBe(501);
    });

    it("returns empty warmupDraws when includeWarmup is false", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      const result = model.sample({
        iterations: 500,
        warmup: 100,
        includeWarmup: false,
        seed: 42n,
      });

      // When includeWarmup is false, warmupDraws is empty array
      expect(result.warmupDraws.length).toBe(0);
    });

    it("returns unconstrained draws when requested", () => {
      const model = defineModel({
        logDensity: (x) => -x[0],
        dim: 1,
        constraints: { 0: "positive" },
      });

      const result = model.sample({
        iterations: 500,
        warmup: 50,
        includeUnconstrained: true,
        seed: 42n,
      });

      expect(result.unconstrainedDraws).toBeDefined();
      expect(result.unconstrainedDraws!.length).toBe(1);
      // Post-warmup draws (MH returns iterations+1 total, warmup takes first 51)
      expect(result.unconstrainedDraws![0].length).toBe(result.draws[0].length);

      // Unconstrained can be any value (positive or negative)
      // Constrained should all be positive
      for (let i = 0; i < result.draws[0].length; i++) {
        expect(result.draws[0][i][0]).toBeGreaterThan(0);
        // Unconstrained is log of constrained for positive transform
        expect(result.unconstrainedDraws![0][i][0]).toBeCloseTo(
          Math.log(result.draws[0][i][0]),
          5,
        );
      }
    });

    it("respects thinning", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      const result = model.sample({
        iterations: 1000,
        warmup: 0,
        thin: 5,
        seed: 42n,
      });

      // MH returns (iterations+1) samples, with thin=5: ceil(1001/5) = 201
      expect(result.draws[0].length).toBe(201);
    });
  });

  describe("summary method", () => {
    it("returns chain summary", () => {
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
      });

      const result = model.sample({
        iterations: 2000,
        warmup: 200,
        chains: 4,
        seed: 42n,
      });

      const summary = result.summary();

      expect(summary.mean).toBeDefined();
      expect(summary.sd).toBeDefined();
      expect(summary.ess).toBeDefined();
      expect(summary.rhat).toBeDefined();

      // For N(0,1), mean should be close to 0
      expect(summary.mean[0]).toBeCloseTo(0, 0);
    });
  });

  describe("distributional correctness", () => {
    it("samples half-normal correctly with positive constraint", () => {
      // Half-normal: p(x) ∝ exp(-x²/2) for x > 0
      // Mean = sqrt(2/π) ≈ 0.798
      const model = defineModel({
        logDensity: (x) => -0.5 * x[0] * x[0],
        dim: 1,
        constraints: { 0: "positive" },
      });

      const result = model.sample({
        iterations: 10000,
        warmup: 1000,
        thin: 5,
        seed: 42n,
      });

      const samples = result.draws[0].map((d) => d[0]);
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      const expectedMean = Math.sqrt(2 / Math.PI);

      expect(mean).toBeCloseTo(expectedMean, 1);
    });

    it("samples Dirichlet correctly with simplex constraint", () => {
      // Dirichlet(2, 3, 4): means = [2/9, 3/9, 4/9]
      const alpha = [2, 3, 4];
      const alphaSum = alpha.reduce((a, b) => a + b, 0);

      const model = defineModel({
        logDensity: (x: Vector) => {
          let logp = 0;
          for (let i = 0; i < x.length; i++) {
            if (x[i] <= 0) return -Infinity;
            logp += (alpha[i] - 1) * Math.log(x[i]);
          }
          return logp;
        },
        dim: 3,
        constraints: { 0: { type: "simplex", k: 3 } },
      });

      const result = model.sample({
        iterations: 15000,
        warmup: 1500,
        thin: 5,
        stepSize: 0.5,
        seed: 456n,
      });

      const samples = result.draws[0];
      const means = [0, 0, 0];
      for (const s of samples) {
        for (let i = 0; i < 3; i++) {
          means[i] += s[i];
        }
      }
      for (let i = 0; i < 3; i++) {
        means[i] /= samples.length;
      }

      const expectedMeans = alpha.map((a) => a / alphaSum);
      for (let i = 0; i < 3; i++) {
        expect(means[i]).toBeCloseTo(expectedMeans[i], 1);
      }
    });
  });
});
