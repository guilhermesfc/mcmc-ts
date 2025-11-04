import { test} from "node:test";
import * as assert from "node:assert/strict";
import { hello } from "../src/index.js";
test("hello works", () => {
  assert.equal(hello(), "mcmc-ts: hello");
});
