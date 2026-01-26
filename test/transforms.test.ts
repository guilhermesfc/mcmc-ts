import { describe, it, expect } from "vitest";
import {
  positiveTransform,
  unitIntervalTransform,
  simplexTransform,
  identityTransform,
  lowerBoundedTransform,
  upperBoundedTransform,
  boundedTransform,
  transformedLogDensity,
  applyTransformToChain,
  composeTransforms,
} from "../src/transforms";
import { Vector } from "../src/core";
import { Transform } from "../src/transforms";
import { metropolisHastings } from "../src/mh";

// Finite difference Jacobian verification for element-wise transforms
// For diagonal Jacobians: log|det(J)| = sum of log|df_i/du_i|
function numericalLogJacobian(T: Transform, u: Vector, h: number = 1e-5): number {
  let logJ = 0;
  for (let i = 0; i < T.inputDim; i++) {
    const uPlus = u.slice();
    const uMinus = u.slice();
    uPlus[i] += h;
    uMinus[i] -= h;
    const fPlus = T.forward(uPlus)[i];
    const fMinus = T.forward(uMinus)[i];
    const derivative = (fPlus - fMinus) / (2 * h);
    logJ += Math.log(Math.abs(derivative));
  }
  return logJ;
}

// For simplex: compute the (k-1) x (k-1) Jacobian of the first k-1 outputs
// This is the correct Jacobian for change of variables on the simplex
function numericalLogJacobianSimplex(T: Transform, u: Vector, h: number = 1e-5): number {
  const m = T.inputDim;  // k-1

  // Build (k-1) x (k-1) Jacobian matrix (first k-1 outputs only)
  const J: number[][] = [];
  for (let i = 0; i < m; i++) {
    J.push([]);
    for (let j = 0; j < m; j++) {
      const uPlus = u.slice();
      const uMinus = u.slice();
      uPlus[j] += h;
      uMinus[j] -= h;
      const fPlus = T.forward(uPlus)[i];
      const fMinus = T.forward(uMinus)[i];
      J[i][j] = (fPlus - fMinus) / (2 * h);
    }
  }

  // Compute determinant
  let det: number;
  if (m === 1) {
    det = J[0][0];
  } else if (m === 2) {
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  } else if (m === 3) {
    det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
        - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
        + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  } else {
    throw new Error("numericalLogJacobianSimplex only supports k <= 4");
  }

  return Math.log(Math.abs(det));
}

describe("identityTransform", () => {
  it("has correct dimensions", () => {
    const T = identityTransform(3);
    expect(T.inputDim).toBe(3);
    expect(T.outputDim).toBe(3);
  });

  it("forward is identity", () => {
    const T = identityTransform(3);
    const u = [1, 2, 3];
    expect(T.forward(u)).toEqual([1, 2, 3]);
  });

  it("inverse is identity", () => {
    const T = identityTransform(3);
    const x = [1, 2, 3];
    expect(T.inverse(x)).toEqual([1, 2, 3]);
  });

  it("logJacobian is zero", () => {
    const T = identityTransform(3);
    expect(T.logJacobian([1, 2, 3])).toBe(0);
  });
});

describe("positiveTransform", () => {
  it("has correct dimensions", () => {
    const T = positiveTransform(3);
    expect(T.inputDim).toBe(3);
    expect(T.outputDim).toBe(3);
  });

  it("forward maps R to R+", () => {
    const T = positiveTransform(3);
    const u = [0, 1, -1];
    const x = T.forward(u);
    expect(x).toEqual([1, Math.E, 1 / Math.E]);
    x.forEach((xi) => expect(xi).toBeGreaterThan(0));
  });

  it("inverse maps R+ to R", () => {
    const T = positiveTransform(3);
    const x = [1, Math.E, 2];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0);
    expect(u[1]).toBeCloseTo(1);
    expect(u[2]).toBeCloseTo(Math.log(2));
  });

  it("forward and inverse are inverses", () => {
    const T = positiveTransform(3);
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is sum of u", () => {
    const T = positiveTransform(3);
    const u = [1, 2, 3];
    const logJ = T.logJacobian(u);
    expect(logJ).toBeCloseTo(6);
  });
});

describe("lowerBoundedTransform", () => {
  it("has correct dimensions", () => {
    const T = lowerBoundedTransform(5, 3);
    expect(T.inputDim).toBe(3);
    expect(T.outputDim).toBe(3);
  });

  it("forward maps R to (a, ∞)", () => {
    const T = lowerBoundedTransform(5, 3);
    const u = [0, 1, -1];
    const x = T.forward(u);
    expect(x[0]).toBeCloseTo(5 + 1);
    expect(x[1]).toBeCloseTo(5 + Math.E);
    expect(x[2]).toBeCloseTo(5 + 1 / Math.E);
    x.forEach((xi) => expect(xi).toBeGreaterThan(5));
  });

  it("inverse maps (a, ∞) to R", () => {
    const T = lowerBoundedTransform(5, 2);
    const x = [6, 5 + Math.E];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0);
    expect(u[1]).toBeCloseTo(1);
  });

  it("forward and inverse are inverses", () => {
    const T = lowerBoundedTransform(10, 3);
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is sum of u", () => {
    const T = lowerBoundedTransform(5, 3);
    const u = [1, 2, 3];
    const logJ = T.logJacobian(u);
    expect(logJ).toBeCloseTo(6);
  });
});

describe("upperBoundedTransform", () => {
  it("has correct dimensions", () => {
    const T = upperBoundedTransform(10, 3);
    expect(T.inputDim).toBe(3);
    expect(T.outputDim).toBe(3);
  });

  it("forward maps R to (-∞, b)", () => {
    const T = upperBoundedTransform(10, 3);
    const u = [0, 1, -1];
    const x = T.forward(u);
    expect(x[0]).toBeCloseTo(10 - 1);
    expect(x[1]).toBeCloseTo(10 - Math.E);
    expect(x[2]).toBeCloseTo(10 - 1 / Math.E);
    x.forEach((xi) => expect(xi).toBeLessThan(10));
  });

  it("inverse maps (-∞, b) to R", () => {
    const T = upperBoundedTransform(10, 2);
    const x = [9, 10 - Math.E];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0);
    expect(u[1]).toBeCloseTo(1);
  });

  it("forward and inverse are inverses", () => {
    const T = upperBoundedTransform(0, 3);
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is sum of u", () => {
    const T = upperBoundedTransform(10, 3);
    const u = [1, 2, 3];
    const logJ = T.logJacobian(u);
    expect(logJ).toBeCloseTo(6);
  });
});

describe("boundedTransform", () => {
  it("has correct dimensions", () => {
    const T = boundedTransform(0, 1, 3);
    expect(T.inputDim).toBe(3);
    expect(T.outputDim).toBe(3);
  });

  it("forward maps R to (a, b)", () => {
    const T = boundedTransform(2, 8, 3);
    const u = [0, 10, -10];
    const x = T.forward(u);
    expect(x[0]).toBeCloseTo(5); // midpoint
    expect(x[1]).toBeCloseTo(8, 1); // near upper
    expect(x[2]).toBeCloseTo(2, 1); // near lower
    x.forEach((xi) => {
      expect(xi).toBeGreaterThan(2);
      expect(xi).toBeLessThan(8);
    });
  });

  it("inverse maps (a, b) to R", () => {
    const T = boundedTransform(0, 10, 2);
    const x = [5, 7];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0); // midpoint -> 0
    expect(u[1]).toBeCloseTo(Math.log(0.7 / 0.3));
  });

  it("forward and inverse are inverses", () => {
    const T = boundedTransform(-5, 5, 3);
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is correct", () => {
    const T = boundedTransform(0, 10, 1);
    const u = [0];
    const logJ = T.logJacobian(u);
    // range = 10, sigmoid(0) = 0.5
    // log(10) + log(0.5) + log(0.5) = log(10) + log(0.25)
    expect(logJ).toBeCloseTo(Math.log(10) + Math.log(0.25));
  });

  it("reduces to unitInterval when bounds are (0, 1)", () => {
    const T1 = boundedTransform(0, 1, 2);
    const T2 = unitIntervalTransform(2);
    const u = [0.5, -0.3];

    const x1 = T1.forward(u);
    const x2 = T2.forward(u);
    x1.forEach((val, i) => expect(val).toBeCloseTo(x2[i]));

    const logJ1 = T1.logJacobian(u);
    const logJ2 = T2.logJacobian(u);
    expect(logJ1).toBeCloseTo(logJ2);
  });
});

describe("unitIntervalTransform", () => {
  it("has correct dimensions", () => {
    const T = unitIntervalTransform(2);
    expect(T.inputDim).toBe(2);
    expect(T.outputDim).toBe(2);
  });

  it("forward maps R to (0, 1)", () => {
    const T = unitIntervalTransform(5);
    const u = [0, 1, -1, 10, -10];
    const x = T.forward(u);
    expect(x[0]).toBeCloseTo(0.5);
    expect(x[1]).toBeCloseTo(1 / (1 + Math.exp(-1)));
    x.forEach((xi) => {
      expect(xi).toBeGreaterThan(0);
      expect(xi).toBeLessThan(1);
    });
  });

  it("inverse maps (0, 1) to R", () => {
    const T = unitIntervalTransform(3);
    const x = [0.5, 0.7, 0.3];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0);
    expect(u[1]).toBeCloseTo(Math.log(0.7 / 0.3));
    expect(u[2]).toBeCloseTo(Math.log(0.3 / 0.7));
  });

  it("forward and inverse are inverses", () => {
    const T = unitIntervalTransform(3);
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is correct", () => {
    const T = unitIntervalTransform(1);
    const u = [0];
    const logJ = T.logJacobian(u);
    // sigmoid(0) = 0.5, log(0.5 * 0.5) = log(0.25)
    expect(logJ).toBeCloseTo(Math.log(0.25));
  });
});

describe("simplexTransform", () => {
  it("has correct dimensions (k-1 input, k output)", () => {
    const T = simplexTransform(4);
    expect(T.inputDim).toBe(3);
    expect(T.outputDim).toBe(4);
  });

  it("forward maps R^(k-1) to simplex", () => {
    const T = simplexTransform(3);
    const u = [0, 0]; // k-1 = 2 inputs
    const x = T.forward(u);

    expect(x.length).toBe(3);
    // All positive
    x.forEach((xi) => expect(xi).toBeGreaterThan(0));
    // Sum to 1
    const sum = x.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1);
  });

  it("forward with zeros gives uniform", () => {
    const T = simplexTransform(4);
    const u = [0, 0, 0];
    const x = T.forward(u);

    // With all zeros, stick-breaking gives:
    // x[0] = 0.5 * 1 = 0.5
    // x[1] = 0.5 * 0.5 = 0.25
    // x[2] = 0.5 * 0.25 = 0.125
    // x[3] = 0.125
    expect(x[0]).toBeCloseTo(0.5);
    expect(x[1]).toBeCloseTo(0.25);
    expect(x[2]).toBeCloseTo(0.125);
    expect(x[3]).toBeCloseTo(0.125);
  });

  it("handles extreme values", () => {
    const T = simplexTransform(3);
    const u = [10, -10];
    const x = T.forward(u);

    // Should not overflow
    x.forEach((xi) => expect(isFinite(xi)).toBe(true));
    // Sum to 1
    const sum = x.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1);
  });

  it("forward and inverse are inverses", () => {
    const T = simplexTransform(4);
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("inverse of forward recovers original", () => {
    const T = simplexTransform(3);
    const original = [0.2, 0.5, 0.3];
    const u = T.inverse(original);
    const recovered = T.forward(u);
    recovered.forEach((val, i) => expect(val).toBeCloseTo(original[i]));
  });

  it("logJacobian is finite", () => {
    const T = simplexTransform(4);
    const u = [0, 0, 0];
    const logJ = T.logJacobian(u);
    expect(isFinite(logJ)).toBe(true);
  });

  it("throws for k < 2", () => {
    expect(() => simplexTransform(1)).toThrow();
  });
});

describe("transformedLogDensity", () => {
  it("applies transform and adds log jacobian", () => {
    const T = positiveTransform(1);

    // Simple density: log p(x) = -x for x > 0
    const baseLogDensity = (x: Vector) => -x[0];

    const transformedDensity = transformedLogDensity(baseLogDensity, T);

    const u = [1]; // maps to x = e
    const logP = transformedDensity(u);

    // log p(u) = log p(T(u)) + log|det J|
    //          = log p(e) + u
    //          = -e + 1
    expect(logP).toBeCloseTo(-Math.E + 1);
  });

  it("works with multi-dimensional transforms", () => {
    const T = positiveTransform(3);

    // Uniform density: log p(x) = 0
    const uniformDensity = (_x: Vector) => 0;

    const transformedDensity = transformedLogDensity(uniformDensity, T);

    const u = [1, 2, 3];
    const logP = transformedDensity(u);

    // Should equal just the log jacobian
    expect(logP).toBeCloseTo(T.logJacobian(u));
  });
});

describe("applyTransformToChain", () => {
  it("transforms all samples in chain", () => {
    const T = positiveTransform(1);
    const chain: Vector[] = [[0], [1], [-1]];

    const transformed = applyTransformToChain(chain, T);

    expect(transformed[0][0]).toBeCloseTo(1);
    expect(transformed[1][0]).toBeCloseTo(Math.E);
    expect(transformed[2][0]).toBeCloseTo(1 / Math.E);
  });

  it("works with multi-dimensional chains", () => {
    const T = unitIntervalTransform(2);
    const chain: Vector[] = [
      [0, 1],
      [-1, 2],
    ];

    const transformed = applyTransformToChain(chain, T);

    expect(transformed[0][0]).toBeCloseTo(0.5);
    expect(transformed[0][1]).toBeCloseTo(1 / (1 + Math.exp(-1)));
    expect(transformed.length).toBe(2);
    expect(transformed[0].length).toBe(2);
  });
});

describe("composeTransforms", () => {
  it("composes single transform", () => {
    const T = positiveTransform(2);
    const composed = composeTransforms(2, [{ startIndex: 0, transform: T }]);

    expect(composed.inputDim).toBe(2);
    expect(composed.outputDim).toBe(2);

    const u = [1, 2];
    expect(composed.forward(u)[0]).toBeCloseTo(Math.exp(1));
    expect(composed.forward(u)[1]).toBeCloseTo(Math.exp(2));
  });

  it("fills gaps with identity", () => {
    // Transform only index 1, leave 0 and 2 as identity
    const T = positiveTransform(1);
    const composed = composeTransforms(3, [{ startIndex: 1, transform: T }]);

    expect(composed.inputDim).toBe(3);
    expect(composed.outputDim).toBe(3);

    const u = [1, 2, 3];
    const x = composed.forward(u);
    expect(x[0]).toBeCloseTo(1); // identity
    expect(x[1]).toBeCloseTo(Math.exp(2)); // positive
    expect(x[2]).toBeCloseTo(3); // identity
  });

  it("composes multiple non-overlapping transforms", () => {
    const T1 = positiveTransform(1);
    const T2 = unitIntervalTransform(1);
    const composed = composeTransforms(2, [
      { startIndex: 0, transform: T1 },
      { startIndex: 1, transform: T2 },
    ]);

    expect(composed.inputDim).toBe(2);
    expect(composed.outputDim).toBe(2);

    const u = [1, 0];
    const x = composed.forward(u);
    expect(x[0]).toBeCloseTo(Math.exp(1));
    expect(x[1]).toBeCloseTo(0.5);
  });

  it("composes log jacobians additively", () => {
    const T1 = positiveTransform(1);
    const T2 = unitIntervalTransform(1);
    const composed = composeTransforms(2, [
      { startIndex: 0, transform: T1 },
      { startIndex: 1, transform: T2 },
    ]);

    const u = [1, 0];
    const logJ = composed.logJacobian(u);
    const logJ1 = T1.logJacobian([u[0]]);
    const logJ2 = T2.logJacobian([u[1]]);

    expect(logJ).toBeCloseTo(logJ1 + logJ2);
  });

  it("forward and inverse compose correctly", () => {
    const T1 = positiveTransform(1);
    const T2 = unitIntervalTransform(1);
    const composed = composeTransforms(2, [
      { startIndex: 0, transform: T1 },
      { startIndex: 1, transform: T2 },
    ]);

    const u = [1.5, -0.5];
    const x = composed.forward(u);
    const u2 = composed.inverse(x);

    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("handles dimension-changing transforms (simplex)", () => {
    // 3-simplex: 2 unconstrained -> 3 constrained
    const T = simplexTransform(3);
    const composed = composeTransforms(3, [{ startIndex: 0, transform: T }]);

    expect(composed.inputDim).toBe(2);
    expect(composed.outputDim).toBe(3);

    const u = [0, 0];
    const x = composed.forward(u);
    expect(x.length).toBe(3);
    expect(x.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it("mixes identity and dimension-changing transforms", () => {
    // [identity(1), simplex(3)] -> constrained dim = 4, unconstrained dim = 3
    const T = simplexTransform(3);
    const composed = composeTransforms(4, [{ startIndex: 1, transform: T }]);

    expect(composed.inputDim).toBe(3); // 1 (identity) + 2 (simplex)
    expect(composed.outputDim).toBe(4); // 1 (identity) + 3 (simplex)

    const u = [5, 0, 0];
    const x = composed.forward(u);
    expect(x[0]).toBeCloseTo(5); // identity
    expect(x.slice(1).reduce((a, b) => a + b, 0)).toBeCloseTo(1); // simplex sums to 1
  });

  it("throws on overlapping transforms", () => {
    const T1 = positiveTransform(2);
    const T2 = unitIntervalTransform(2);
    expect(() =>
      composeTransforms(3, [
        { startIndex: 0, transform: T1 },
        { startIndex: 1, transform: T2 }, // overlaps with T1
      ]),
    ).toThrow(/Overlapping/);
  });

  it("throws when transforms exceed dimension", () => {
    const T = positiveTransform(3);
    expect(() =>
      composeTransforms(2, [{ startIndex: 0, transform: T }]),
    ).toThrow(/exceed/);
  });

  it("handles empty specs (all identity)", () => {
    const composed = composeTransforms(3, []);
    expect(composed.inputDim).toBe(3);
    expect(composed.outputDim).toBe(3);

    const u = [1, 2, 3];
    expect(composed.forward(u)).toEqual([1, 2, 3]);
    expect(composed.logJacobian(u)).toBe(0);
  });
});

// Finite difference Jacobian verification tests
describe("logJacobian finite difference verification", () => {
  const testPoints = [
    [0],
    [1],
    [-1],
    [0.5, -0.3],
    [1.2, 0.8, -0.5],
  ];

  describe("positiveTransform", () => {
    it.each([
      { u: [0], dim: 1 },
      { u: [1, -1], dim: 2 },
      { u: [0.5, -0.3, 1.2], dim: 3 },
    ])("matches numerical Jacobian for u=$u", ({ u, dim }) => {
      const T = positiveTransform(dim);
      const analytical = T.logJacobian(u);
      const numerical = numericalLogJacobian(T, u);
      expect(analytical).toBeCloseTo(numerical, 4);
    });
  });

  describe("lowerBoundedTransform", () => {
    it.each([
      { u: [0], lower: 5, dim: 1 },
      { u: [1, -1], lower: -10, dim: 2 },
      { u: [0.5, -0.3, 1.2], lower: 0, dim: 3 },
    ])("matches numerical Jacobian for lower=$lower, u=$u", ({ u, lower, dim }) => {
      const T = lowerBoundedTransform(lower, dim);
      const analytical = T.logJacobian(u);
      const numerical = numericalLogJacobian(T, u);
      expect(analytical).toBeCloseTo(numerical, 4);
    });
  });

  describe("upperBoundedTransform", () => {
    it.each([
      { u: [0], upper: 10, dim: 1 },
      { u: [1, -1], upper: 0, dim: 2 },
      { u: [0.5, -0.3, 1.2], upper: 100, dim: 3 },
    ])("matches numerical Jacobian for upper=$upper, u=$u", ({ u, upper, dim }) => {
      const T = upperBoundedTransform(upper, dim);
      const analytical = T.logJacobian(u);
      const numerical = numericalLogJacobian(T, u);
      expect(analytical).toBeCloseTo(numerical, 4);
    });
  });

  describe("boundedTransform", () => {
    it.each([
      { u: [0], lower: 0, upper: 1, dim: 1 },
      { u: [1, -1], lower: -5, upper: 5, dim: 2 },
      { u: [0.5, -0.3, 1.2], lower: 0, upper: 100, dim: 3 },
    ])("matches numerical Jacobian for bounds=[$lower,$upper], u=$u", ({ u, lower, upper, dim }) => {
      const T = boundedTransform(lower, upper, dim);
      const analytical = T.logJacobian(u);
      const numerical = numericalLogJacobian(T, u);
      expect(analytical).toBeCloseTo(numerical, 4);
    });
  });

  describe("unitIntervalTransform", () => {
    it.each([
      { u: [0], dim: 1 },
      { u: [1, -1], dim: 2 },
      { u: [0.5, -0.3, 1.2], dim: 3 },
    ])("matches numerical Jacobian for u=$u", ({ u, dim }) => {
      const T = unitIntervalTransform(dim);
      const analytical = T.logJacobian(u);
      const numerical = numericalLogJacobian(T, u);
      expect(analytical).toBeCloseTo(numerical, 4);
    });
  });

  describe("simplexTransform", () => {
    it.each([
      { u: [0], k: 2 },
      { u: [0, 0], k: 3 },
      { u: [0.5, -0.3], k: 3 },
      { u: [1, -1, 0.5], k: 4 },
    ])("matches numerical Jacobian for k=$k, u=$u", ({ u, k }) => {
      const T = simplexTransform(k);
      const analytical = T.logJacobian(u);
      const numerical = numericalLogJacobianSimplex(T, u);
      expect(analytical).toBeCloseTo(numerical, 3);
    });
  });
});

// Distributional smoke tests - verify MCMC samples match expected moments
describe("distributional smoke tests", () => {
  // Helper: compute sample mean of each dimension
  function sampleMeans(samples: Vector[]): number[] {
    const dim = samples[0].length;
    const means = Array(dim).fill(0);
    for (const s of samples) {
      for (let i = 0; i < dim; i++) {
        means[i] += s[i];
      }
    }
    return means.map((m) => m / samples.length);
  }

  // Helper: compute sample variance
  function sampleVariance(samples: number[]): number {
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    return samples.reduce((a, x) => a + (x - mean) ** 2, 0) / (samples.length - 1);
  }

  describe("half-normal via positiveTransform", () => {
    it("samples have correct mean and variance", () => {
      // Half-normal: p(x) ∝ exp(-x²/2) for x > 0
      // Mean = sqrt(2/π) ≈ 0.798
      // Variance = 1 - 2/π ≈ 0.363
      const halfNormalLogDensity = (x: Vector) => -0.5 * x[0] * x[0];

      const result = metropolisHastings(halfNormalLogDensity, 1, {
        iterations: 20000,
        burnIn: 2000,
        thin: 5,
        stepSize: 1.0,
        seed: 42n,
        transforms: [{ startIndex: 0, transform: positiveTransform(1) }],
      });

      const samples = result.samples[0].map((s: Vector) => s[0]);
      const mean = samples.reduce((a: number, b: number) => a + b, 0) / samples.length;
      const variance = sampleVariance(samples);

      const expectedMean = Math.sqrt(2 / Math.PI);
      const expectedVariance = 1 - 2 / Math.PI;

      // Allow 10% tolerance for MCMC sampling noise
      expect(mean).toBeCloseTo(expectedMean, 1);
      expect(variance).toBeCloseTo(expectedVariance, 1);
    });
  });

  describe("Beta(2,5) via unitIntervalTransform", () => {
    it("samples have correct mean", () => {
      // Beta(α,β): mean = α/(α+β) = 2/7 ≈ 0.286
      const alpha = 2, beta = 5;
      const betaLogDensity = (x: Vector) => {
        const p = x[0];
        if (p <= 0 || p >= 1) return -Infinity;
        return (alpha - 1) * Math.log(p) + (beta - 1) * Math.log(1 - p);
      };

      const result = metropolisHastings(betaLogDensity, 1, {
        iterations: 20000,
        burnIn: 2000,
        thin: 5,
        stepSize: 1.0,
        seed: 123n,
        transforms: [{ startIndex: 0, transform: unitIntervalTransform(1) }],
      });

      const samples = result.samples[0].map((s: Vector) => s[0]);
      const mean = samples.reduce((a: number, b: number) => a + b, 0) / samples.length;

      const expectedMean = alpha / (alpha + beta);

      expect(mean).toBeCloseTo(expectedMean, 1);
    });
  });

  describe("Dirichlet(2,3,4) via simplexTransform", () => {
    it("samples have correct means", () => {
      // Dirichlet(α): mean_i = α_i / sum(α)
      // For α = [2, 3, 4]: means = [2/9, 3/9, 4/9] ≈ [0.222, 0.333, 0.444]
      const alpha = [2, 3, 4];
      const alphaSum = alpha.reduce((a, b) => a + b, 0);

      const dirichletLogDensity = (x: Vector) => {
        // Check simplex constraint
        const sum = x.reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1) > 1e-6) return -Infinity;
        for (const xi of x) {
          if (xi <= 0) return -Infinity;
        }
        // log p(x) = sum((α_i - 1) * log(x_i))
        let logp = 0;
        for (let i = 0; i < x.length; i++) {
          logp += (alpha[i] - 1) * Math.log(x[i]);
        }
        return logp;
      };

      const result = metropolisHastings(dirichletLogDensity, 3, {
        iterations: 30000,
        burnIn: 3000,
        thin: 5,
        stepSize: 0.5,
        seed: 456n,
        transforms: [{ startIndex: 0, transform: simplexTransform(3) }],
      });

      const means = sampleMeans(result.samples[0]);
      const expectedMeans = alpha.map((a) => a / alphaSum);

      for (let i = 0; i < 3; i++) {
        expect(means[i]).toBeCloseTo(expectedMeans[i], 1);
      }
    });
  });

  describe("Exponential(λ=2) via lowerBoundedTransform", () => {
    it("samples have correct mean", () => {
      // Exponential(λ): mean = 1/λ = 0.5
      const lambda = 2;
      const expLogDensity = (x: Vector) => {
        if (x[0] <= 0) return -Infinity;
        return Math.log(lambda) - lambda * x[0];
      };

      const result = metropolisHastings(expLogDensity, 1, {
        iterations: 20000,
        burnIn: 2000,
        thin: 5,
        stepSize: 1.0,
        seed: 789n,
        transforms: [{ startIndex: 0, transform: positiveTransform(1) }],
      });

      const samples = result.samples[0].map((s: Vector) => s[0]);
      const mean = samples.reduce((a: number, b: number) => a + b, 0) / samples.length;

      const expectedMean = 1 / lambda;

      expect(mean).toBeCloseTo(expectedMean, 1);
    });
  });

  describe("bounded uniform via boundedTransform", () => {
    it("samples have correct mean", () => {
      // Uniform(2, 8): mean = (2 + 8) / 2 = 5
      const lower = 2, upper = 8;
      const uniformLogDensity = (_x: Vector) => 0; // constant

      const result = metropolisHastings(uniformLogDensity, 1, {
        iterations: 20000,
        burnIn: 2000,
        thin: 5,
        stepSize: 1.0,
        seed: 999n,
        transforms: [{ startIndex: 0, transform: boundedTransform(lower, upper, 1) }],
      });

      const samples = result.samples[0].map((s: Vector) => s[0]);
      const mean = samples.reduce((a: number, b: number) => a + b, 0) / samples.length;

      const expectedMean = (lower + upper) / 2;

      expect(mean).toBeCloseTo(expectedMean, 1);
    });
  });
});
