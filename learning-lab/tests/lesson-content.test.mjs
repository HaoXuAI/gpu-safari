import test from "node:test";
import assert from "node:assert/strict";

import {
  CODE_SAMPLES,
  getLessonCopy,
  getPredictionFeedback,
} from "../src/lesson-content.mjs";

test("every lesson step has concise progressive-reveal copy", () => {
  for (const step of [
    "story",
    "predict",
    "simulate",
    "code",
    "run",
    "explain",
    "challenge",
  ]) {
    const copy = getLessonCopy(step);

    assert.equal(typeof copy.eyebrow, "string");
    assert.equal(typeof copy.title, "string");
    assert.ok(copy.eyebrow.length > 0);
    assert.ok(copy.title.length > 0);
  }
});

test("prediction feedback rewards reasoning without claiming measured speed", () => {
  assert.match(getPredictionFeedback("gpu"), /parallel/i);
  assert.match(getPredictionFeedback("cpu"), /reasonable/i);
  assert.doesNotMatch(getPredictionFeedback("gpu"), /milliseconds|faster by/i);
});

test("the Triton sample teaches block programs and masked vector stores", () => {
  assert.match(CODE_SAMPLES.triton, /tl\.program_id\(0\)/);
  assert.match(CODE_SAMPLES.triton, /tl\.arange/);
  assert.match(CODE_SAMPLES.triton, /tl\.store/);
  assert.match(CODE_SAMPLES.triton, /mask=/);
  assert.doesNotMatch(CODE_SAMPLES.triton, /if pixel|image\[pixel\]/);
});
