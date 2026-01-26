import { LogDensity, Vector } from "./core.js";
import { sampleMH } from "./mh.js";
import type {
  BuiltInModel,
  WorkerRequest,
  WorkerResponse,
} from "./worker-types.js";

/**
 * Convert a declarative model spec into a logDensity function.
 * This runs inside the worker where we CAN have functions.
 */
function modelToLogDensity(model: BuiltInModel, dim: number): LogDensity {
  switch (model.type) {
    case "normal": {
      // Univariate normal: log p(x) = -0.5 * ((x - mean) / std)^2 + const
      const { mean, std } = model;
      const variance = std * std;
      return (x: Vector) => {
        let sum = 0;
        for (let i = 0; i < dim; i++) {
          const diff = x[i] - mean;
          sum += (diff * diff) / variance;
        }
        return -0.5 * sum;
      };
    }

    case "mvnormal": {
      // Multivariate normal with diagonal covariance (simplified)
      // For full covariance, you'd need matrix inversion
      const { mean, cov } = model;
      // Extract diagonal (assume diagonal covariance for simplicity)
      const variances = cov.map((row, i) => row[i]);
      return (x: Vector) => {
        let sum = 0;
        for (let i = 0; i < dim; i++) {
          const diff = x[i] - mean[i];
          sum += (diff * diff) / variances[i];
        }
        return -0.5 * sum;
      };
    }

    case "bimodal": {
      // Mixture of two 2D Gaussians separated along x-axis
      const { sep } = model;
      const halfSep = sep / 2;
      return (x: Vector) => {
        // Mode 1 at (-halfSep, 0), Mode 2 at (+halfSep, 0)
        const d1 = (x[0] + halfSep) * (x[0] + halfSep) + x[1] * x[1];
        const d2 = (x[0] - halfSep) * (x[0] - halfSep) + x[1] * x[1];
        // log-sum-exp for numerical stability
        const max = Math.max(-0.5 * d1, -0.5 * d2);
        return (
          max + Math.log(Math.exp(-0.5 * d1 - max) + Math.exp(-0.5 * d2 - max))
        );
      };
    }

    case "banana": {
      // Rosenbrock/banana-shaped distribution
      // p(x,y) ∝ exp(-a*x² - b*(y - x²)²)
      const { a, b } = model;
      return (x: Vector) => {
        const term1 = a * x[0] * x[0];
        const term2 = b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
        return -0.5 * (term1 + term2);
      };
    }

    default:
      throw new Error(`Unknown model type: ${(model as BuiltInModel).type}`);
  }
}

/**
 * Post a typed response back to the main thread.
 */
function respond(response: WorkerResponse): void {
  self.postMessage(response);
}

/**
 * Handle incoming sample request.
 */
function handleRequest(request: WorkerRequest): void {
  try {
    const { model, dim, options } = request;
    const logDensity = modelToLogDensity(model, dim);

    // Parse seed back to bigint
    const seed = options.seed ? BigInt(options.seed) : 0n;

    const gen = sampleMH(logDensity, dim, {
      ...options,
      seed, // <-- use parsed bigint
      yieldEvery: options.iterations > 1000 ? 100 : 10,
    });

    let result = gen.next();
    while (!result.done) {
      // Post progress snapshot
      respond({ type: "progress", snapshot: result.value });
      result = gen.next();
    }

    // Post final results
    const { samples, acceptanceRates } = result.value;
    respond({ type: "complete", samples, acceptanceRates });
  } catch (err) {
    respond({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}

// Listen for messages from main thread
self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  handleRequest(event.data);
};
