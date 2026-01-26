import { Vector } from "./core.js";
import { LogDensity } from "./core";

export interface Transform {
  inputDim: number; // unconstrained space dimension
  outputDim: number; // constrained space dimension
  forward(u: Vector): Vector; // R^inputDim -> R^outputDim
  inverse(x: Vector): Vector; // R^outputDim -> R^inputDim
  logJacobian(u: Vector): number;
}

export interface TransformSpec {
  startIndex: number; // where this block starts in constrained space
  transform: Transform;
}

// Identity transform for unconstrained dimensions
export function identityTransform(dim: number = 1): Transform {
  return {
    inputDim: dim,
    outputDim: dim,
    forward(u) {
      return u.slice();
    },
    inverse(x) {
      return x.slice();
    },
    logJacobian(_u) {
      return 0;
    },
  };
}

// Compose transforms with sparse specification and identity fill
export function composeTransforms(
  constrainedDim: number,
  specs: TransformSpec[],
): Transform {
  // Sort specs by startIndex
  const sorted = [...specs].sort((a, b) => a.startIndex - b.startIndex);

  // Validate no overlaps and build full block list
  const blocks: { startConstrained: number; transform: Transform }[] = [];
  let cursor = 0;

  for (const spec of sorted) {
    if (spec.startIndex < cursor) {
      throw new Error(
        `Overlapping transforms at index ${spec.startIndex} (previous block ends at ${cursor})`,
      );
    }
    // Fill gap with identity if needed
    if (spec.startIndex > cursor) {
      const gapSize = spec.startIndex - cursor;
      blocks.push({ startConstrained: cursor, transform: identityTransform(gapSize) });
    }
    blocks.push({ startConstrained: spec.startIndex, transform: spec.transform });
    cursor = spec.startIndex + spec.transform.outputDim;
  }

  // Fill trailing gap with identity if needed
  if (cursor < constrainedDim) {
    blocks.push({ startConstrained: cursor, transform: identityTransform(constrainedDim - cursor) });
  } else if (cursor > constrainedDim) {
    throw new Error(
      `Transforms exceed constrainedDim: blocks end at ${cursor}, expected ${constrainedDim}`,
    );
  }

  // Compute total input dimension
  const totalInputDim = blocks.reduce((sum, b) => sum + b.transform.inputDim, 0);

  return {
    inputDim: totalInputDim,
    outputDim: constrainedDim,
    forward(u) {
      const out: number[] = [];
      let uCursor = 0;
      for (const block of blocks) {
        const slice = u.slice(uCursor, uCursor + block.transform.inputDim);
        out.push(...block.transform.forward(slice));
        uCursor += block.transform.inputDim;
      }
      return out;
    },
    inverse(x) {
      const out: number[] = [];
      let xCursor = 0;
      for (const block of blocks) {
        const slice = x.slice(xCursor, xCursor + block.transform.outputDim);
        out.push(...block.transform.inverse(slice));
        xCursor += block.transform.outputDim;
      }
      return out;
    },
    logJacobian(u) {
      let total = 0;
      let uCursor = 0;
      for (const block of blocks) {
        const slice = u.slice(uCursor, uCursor + block.transform.inputDim);
        total += block.transform.logJacobian(slice);
        uCursor += block.transform.inputDim;
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
  return chain.map((u) => T.forward(u));
}

// TRANSFORM IMPLEMENTATIONS

// Lower Bounded Transform: u -> x, where x = a + exp(u)
export function lowerBoundedTransform(lower: number, dim: number = 1): Transform {
  return {
    inputDim: dim,
    outputDim: dim,
    forward(u) {
      return u.map((ui) => lower + Math.exp(ui));
    },
    inverse(x) {
      return x.map((xi) => Math.log(xi - lower));
    },
    logJacobian(u) {
      return u.reduce((a, ui) => a + ui, 0);
    },
  };
}

// Upper Bounded Transform: u -> x, where x = b - exp(u)
export function upperBoundedTransform(upper: number, dim: number = 1): Transform {
  return {
    inputDim: dim,
    outputDim: dim,
    forward(u) {
      return u.map((ui) => upper - Math.exp(ui));
    },
    inverse(x) {
      return x.map((xi) => Math.log(upper - xi));
    },
    logJacobian(u) {
      return u.reduce((a, ui) => a + ui, 0);
    },
  };
}

// Bounded Transform: u -> x, where x = a + (b - a) * sigmoid(u)
export function boundedTransform(lower: number, upper: number, dim: number = 1): Transform {
  const range = upper - lower;
  return {
    inputDim: dim,
    outputDim: dim,
    forward(u) {
      return u.map((ui) => lower + range * sigmoid(ui));
    },
    inverse(x) {
      return x.map((xi) => logit((xi - lower) / range));
    },
    logJacobian(u) {
      return u.reduce((a, ui) => {
        const s = sigmoid(ui);
        return a + Math.log(range) + Math.log(s) + Math.log(1 - s);
      }, 0);
    },
  };
}

// Positive Transform: u -> x, where x = exp(u)
export function positiveTransform(dim: number = 1): Transform {
  return {
    inputDim: dim,
    outputDim: dim,
    forward(u) {
      return u.map(Math.exp);
    },
    inverse(x) {
      return x.map(Math.log);
    },
    logJacobian(u) {
      return u.reduce((a, ui) => a + ui, 0);
    },
  };
}

// Unit Interval Transform: u -> x, where x = sigmoid(u)
function logit(p: number): number {
  return Math.log(p / (1 - p));
}
function sigmoid(z: number) {
  return 1 / (1 + Math.exp(-z));
}

export function unitIntervalTransform(dim: number = 1): Transform {
  return {
    inputDim: dim,
    outputDim: dim,
    forward(u) {
      return u.map(sigmoid);
    },
    inverse(x) {
      return x.map(logit);
    },
    logJacobian(u) {
      return u.map(sigmoid).reduce((a, s) => a + Math.log(s * (1 - s)), 0);
    },
  };
}

// Simplex Transform: u ∈ R^(k-1) -> x ∈ Δ^k (k-simplex)
// Uses stick-breaking parameterization
export function simplexTransform(k: number): Transform {
  if (k < 2) {
    throw new Error("simplexTransform requires k >= 2");
  }
  return {
    inputDim: k - 1,
    outputDim: k,
    forward(u) {
      // Stick-breaking: x_i = sigmoid(u_i) * remaining
      const x: number[] = [];
      let remaining = 1;
      for (let i = 0; i < k - 1; i++) {
        const p = sigmoid(u[i]);
        x.push(p * remaining);
        remaining *= 1 - p;
      }
      x.push(remaining);
      return x;
    },
    inverse(x) {
      // Inverse stick-breaking
      const u: number[] = [];
      let remaining = 1;
      for (let i = 0; i < k - 1; i++) {
        const p = x[i] / remaining;
        u.push(logit(p));
        remaining -= x[i];
      }
      return u;
    },
    logJacobian(u) {
      // Product of sigmoid derivatives times remaining mass
      let logJ = 0;
      let remaining = 1;
      for (let i = 0; i < k - 1; i++) {
        const s = sigmoid(u[i]);
        // d/du_i of (s * remaining) = s*(1-s) * remaining
        logJ += Math.log(s * (1 - s)) + Math.log(remaining);
        remaining *= 1 - s;
      }
      return logJ;
    },
  };
}
