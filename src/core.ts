export type Vector = number[];
export type LogDensity = (x: Vector) => number;

export interface RNG {
  normal(mean?: number, std?: number): number;
  uniform(): number;
}

// PCG-XSH-RR 32-bit-wide Xorshift-based RNG.
// Based on:
// O’Neill, Melissa E. (2014), "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Random Number Generators".
export class PCG32 implements RNG {
  private state: bigint;
  private inc: bigint;
  private spare: number | null = null;

  constructor(seed: bigint = 123n, seq: bigint = 0n) {
    // independent stream via seq (odd increment = 2*seq+1)
    this.state = 0n;
    this.inc = (seq << 1n) | 1n;
    this.nextU32(); // scramble
    this.state += seed;
    this.nextU32(); // scramble again
  }

  private nextU32(): number {
    // state = state * 6364136223846793005 + inc (mod 2^64)
    this.state =
      (this.state * 6364136223846793005n + this.inc) & ((1n << 64n) - 1n);
    const xorshifted = Number(((this.state >> 18n) ^ this.state) >> 27n) >>> 0;
    const rot = Number(this.state >> 59n) & 31;
    return ((xorshifted >>> rot) | (xorshifted << ((-rot >>> 0) & 31))) >>> 0;
  }

  // ~53-bit uniform in [0,1)
  uniform(): number {
    const a = this.nextU32(); // 32 bits
    const b = this.nextU32(); // 32 bits
    const hi = a >>> 5; // 27 bits
    const lo = b >>> 6; // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992; // 2^53
  }

  normal(mean = 0, std = 1): number {
    if (this.spare !== null) {
      const z1 = this.spare;
      this.spare = null;
      return mean + std * z1;
    }
    let u = 0;
    do {
      u = this.uniform();
    } while (u === 0); // keep log defined
    const v = this.uniform();
    const mag = Math.sqrt(-2 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    this.spare = z1;
    return mean + std * z0;
  }
}

export function add(a: Vector, b: Vector): Vector {
  if (a.length !== b.length) throw new Error("Dimension mismatch");
  return a.map((ai, i) => ai + b[i]);
}

export function scale(a: Vector, s: number): Vector {
  return a.map((ai) => ai * s);
}

export function zeros(d: number): Vector {
  return Array(d).fill(0);
}
