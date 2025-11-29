import { Vector } from "./core.js";
import { LogDensity } from "./core";

export interface Transform {
  // from unconstrained u ∈ R^d -> constrained x
  forward(u: Vector): Vector;

  // inverse: x -> u, only needed if user passes constrained start values
  inverse(x: Vector): Vector;

  // log |det J| of the forward transform
  logJacobian(u: Vector): number;
}

// Compose Transforms: T1 ∘ T2 ∘ ... ∘ Tn
export function composeTransforms(ts: Transform[]): Transform {
  return {
    forward(u) {
      const out: number[] = [];
      let i = 0;
      for (const T of ts) {
        const slice = [u[i]];
        out.push(...T.forward(slice));
        i++;
      }
      return out;
    },
    inverse(x) {
      const out: number[] = [];
      let i = 0;
      for (const T of ts) {
        const slice = [x[i]];
        out.push(...T.inverse(slice));
        i++;
      }
      return out;
    },
    logJacobian(u) {
      let total = 0;
      let i = 0;
      for (const T of ts) {
        total += T.logJacobian([u[i]]);
        i++;
      }
      return total;
    },
  };
}

export function transformedLogDensity(
  baseLogDensity: LogDensity,
  T: Transform,
): LogDensity {
  return (u: Vector) => {
    const x = T.forward(u);
    return baseLogDensity(x) + T.logJacobian(u);
  };
}

export function applyTransformToChain(chain: Vector[], T: Transform): Vector[] {
  return chain.map(T.forward);
}

// TRANSFORMS IMPLEMENTATIONS

// Positive Transform: u -> x, where x = exp(u)
export function positiveTransform(): Transform {
  return {
    forward(u) {
      return u.map(Math.exp);
    },
    inverse(x) {
      return x.map(Math.log);
    },
    logJacobian(u) {
      // det(J) = exp(sum(u))
      return u.reduce((a, ui) => a + ui, 0);
    },
  };
}

// Logit Transform: p -> z, where z = log(p / (1 - p))
function logit(p: number): number {
  return Math.log(p / (1 - p));
}
function sigmoid(z: number) {
  return 1 / (1 + Math.exp(-z));
}

export function unitIntervalTransform(): Transform {
  return {
    forward(u) {
      return u.map(sigmoid);
    },
    inverse(x) {
      return x.map(logit);
    },
    logJacobian(u) {
      // derivative: sigmoid(u)*(1-sigmoid(u))
      return u.map(sigmoid).reduce((a, s) => a + Math.log(s * (1 - s)), 0);
    },
  };
}

// Simplex Transform: u -> x, where x is a probability vector
export function simplexTransform(): Transform {
  return {
    forward(u) {
      const m = Math.max(...u);
      const exps = u.map((ui) => Math.exp(ui - m));
      const Z = exps.reduce((a, e) => a + e, 0);
      return exps.map((e) => e / Z);
    },
    inverse(x) {
      // inverse softmax: log(x_i) - log(x_k)
      const k = x.length;
      return x
        .slice(0, k - 1)
        .map((xi, i) => Math.log(xi) - Math.log(x[k - 1]));
    },
    logJacobian(u) {
      // logJ = sum(u - logsumexp(u))
      const m = Math.max(...u);
      const sumExp = u.reduce((a, ui) => a + Math.exp(ui - m), 0);
      const L = m + Math.log(sumExp);
      return u.reduce((a, ui) => a + (ui - L), 0);
    },
  };
}
