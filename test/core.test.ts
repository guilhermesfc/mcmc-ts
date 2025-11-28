import { describe, it, expect } from "vitest";
import { PCG32, add, scale, zeros, Vector } from "../src/core";

describe("PCG32", () => {
  it("generates uniform values in [0, 1)", () => {
    const rng = new PCG32(42n);
    for (let i = 0; i < 100; i++) {
      const u = rng.uniform();
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
    }
  });

  it("generates reproducible sequences with same seed", () => {
    const rng1 = new PCG32(123n);
    const rng2 = new PCG32(123n);
    for (let i = 0; i < 10; i++) {
      expect(rng1.uniform()).toBe(rng2.uniform());
    }
  });

  it("generates normal samples with correct mean and std", () => {
    const rng = new PCG32(42n);
    const samples: Vector = [];
    for (let i = 0; i < 10000; i++) {
      samples.push(rng.normal(6, 3));
    }
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance =
      samples.reduce((s, x) => s + (x - mean) ** 2, 0) / samples.length;
    const std = Math.sqrt(variance);

    expect(mean).toBeCloseTo(6, 0);
    expect(std).toBeCloseTo(3, 0);
  });
});

describe("vector operations", () => {
  it("adds vectors element-wise", () => {
    expect(add([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it("throws on dimension mismatch", () => {
    expect(() => add([1, 2], [1, 2, 3])).toThrow("Dimension mismatch");
  });

  it("scales vectors", () => {
    expect(scale([1, 2, 3], 2)).toEqual([2, 4, 6]);
  });

  it("creates zero vectors", () => {
    expect(zeros(3)).toEqual([0, 0, 0]);
  });
});
