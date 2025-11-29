import { describe, it, expect } from "vitest";
import {
  positiveTransform,
  unitIntervalTransform,
  simplexTransform,
  transformedLogDensity,
  applyTransformToChain,
  composeTransforms,
} from "../src/transforms";
import { Vector } from "../src/core";

describe("positiveTransform", () => {
  const T = positiveTransform();

  it("forward maps R to R+", () => {
    const u = [0, 1, -1];
    const x = T.forward(u);
    expect(x).toEqual([1, Math.E, 1 / Math.E]);
    x.forEach((xi) => expect(xi).toBeGreaterThan(0));
  });

  it("inverse maps R+ to R", () => {
    const x = [1, Math.E, 2];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0);
    expect(u[1]).toBeCloseTo(1);
    expect(u[2]).toBeCloseTo(Math.log(2));
  });

  it("forward and inverse are inverses", () => {
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is sum of u", () => {
    const u = [1, 2, 3];
    const logJ = T.logJacobian(u);
    expect(logJ).toBeCloseTo(6);
  });
});

describe("unitIntervalTransform", () => {
  const T = unitIntervalTransform();

  it("forward maps R to (0, 1)", () => {
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
    const x = [0.5, 0.7, 0.3];
    const u = T.inverse(x);
    expect(u[0]).toBeCloseTo(0);
    expect(u[1]).toBeCloseTo(Math.log(0.7 / 0.3));
    expect(u[2]).toBeCloseTo(Math.log(0.3 / 0.7));
  });

  it("forward and inverse are inverses", () => {
    const u = [0.5, -0.3, 1.2];
    const x = T.forward(u);
    const u2 = T.inverse(x);
    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });

  it("logJacobian is correct", () => {
    const u = [0];
    const logJ = T.logJacobian(u);
    // sigmoid(0) = 0.5, log(0.5 * 0.5) = log(0.25)
    expect(logJ).toBeCloseTo(Math.log(0.25));
  });
});

describe("simplexTransform", () => {
  const T = simplexTransform();

  it("forward maps R^(k-1) to simplex", () => {
    const u = [0, 0, 0];
    const x = T.forward(u);

    // All positive
    x.forEach((xi) => expect(xi).toBeGreaterThan(0));

    // Sum to 1
    const sum = x.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1);
  });

  it("forward is softmax", () => {
    const u = [1, 2, 3];
    const x = T.forward(u);

    const expSum = Math.exp(1) + Math.exp(2) + Math.exp(3);
    expect(x[0]).toBeCloseTo(Math.exp(1) / expSum);
    expect(x[1]).toBeCloseTo(Math.exp(2) / expSum);
    expect(x[2]).toBeCloseTo(Math.exp(3) / expSum);
  });

  it("handles large values numerically", () => {
    const u = [100, 101, 102];
    const x = T.forward(u);

    // Should not overflow
    x.forEach((xi) => expect(isFinite(xi)).toBe(true));

    // Sum to 1
    const sum = x.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1);
  });

  it("inverse returns k-1 dimensions", () => {
    const x = [0.2, 0.5, 0.3];
    const u = T.inverse(x);

    expect(u.length).toBe(2); // k-1
    expect(u[0]).toBeCloseTo(Math.log(0.2) - Math.log(0.3));
    expect(u[1]).toBeCloseTo(Math.log(0.5) - Math.log(0.3));
  });

  it("logJacobian is sum(u - logsumexp(u))", () => {
    const u = [1, 2, 3];
    const logJ = T.logJacobian(u);

    const m = 3; // max
    const sumExp = Math.exp(1 - 3) + Math.exp(2 - 3) + Math.exp(3 - 3);
    const L = 3 + Math.log(sumExp);
    const expected = 1 - L + (2 - L) + (3 - L);

    expect(logJ).toBeCloseTo(expected);
  });
});

describe("transformedLogDensity", () => {
  it("applies transform and adds log jacobian", () => {
    const T = positiveTransform();

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
    const T = simplexTransform();

    // Uniform on simplex: log p(x) = 0
    const uniformDensity = (x: Vector) => 0;

    const transformedDensity = transformedLogDensity(uniformDensity, T);

    const u = [0, 0, 0];
    const logP = transformedDensity(u);

    // Should equal just the log jacobian
    expect(logP).toBeCloseTo(T.logJacobian(u));
  });
});

describe("applyTransformToChain", () => {
  it("transforms all samples in chain", () => {
    const T = positiveTransform();
    const chain: Vector[] = [[0], [1], [-1]];

    const transformed = applyTransformToChain(chain, T);

    expect(transformed[0][0]).toBeCloseTo(1);
    expect(transformed[1][0]).toBeCloseTo(Math.E);
    expect(transformed[2][0]).toBeCloseTo(1 / Math.E);
  });

  it("works with multi-dimensional chains", () => {
    const T = unitIntervalTransform();
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
  it("composes transforms element-wise", () => {
    const T1 = positiveTransform();
    const T2 = positiveTransform();

    const composed = composeTransforms([T1, T2]);

    const u = [1, 2];
    const x = composed.forward(u);

    // Each transform operates on one element
    expect(x[0]).toBeCloseTo(Math.exp(1));
    expect(x[1]).toBeCloseTo(Math.exp(2));
  });

  it("composes log jacobians additively", () => {
    const T1 = positiveTransform();
    const T2 = unitIntervalTransform();

    const composed = composeTransforms([T1, T2]);

    const u = [1, 0];
    const logJ = composed.logJacobian(u);

    // Should be sum of individual jacobians
    const logJ1 = T1.logJacobian([u[0]]);
    const logJ2 = T2.logJacobian([u[1]]);

    expect(logJ).toBeCloseTo(logJ1 + logJ2);
  });

  it("forward and inverse compose correctly", () => {
    const T1 = positiveTransform();
    const T2 = unitIntervalTransform();

    const composed = composeTransforms([T1, T2]);

    const u = [1.5, -0.5];
    const x = composed.forward(u);
    const u2 = composed.inverse(x);

    u2.forEach((val, i) => expect(val).toBeCloseTo(u[i]));
  });
});
