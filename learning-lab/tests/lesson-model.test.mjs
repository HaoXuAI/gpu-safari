import test from "node:test";
import assert from "node:assert/strict";

import {
  LESSON_STEPS,
  advanceLesson,
  buildPixelWork,
  createLessonState,
  recordPrediction,
  retreatLesson,
  nextGridIndex,
} from "../src/lesson-model.mjs";

test("a new lesson starts at the story with no prediction", () => {
  const state = createLessonState();

  assert.deepEqual(LESSON_STEPS, [
    "story",
    "predict",
    "simulate",
    "code",
    "run",
    "explain",
    "challenge",
  ]);
  assert.equal(state.stepIndex, 0);
  assert.equal(state.prediction, null);
  assert.deepEqual(state.completed, []);
});

test("advancing returns a new state and records the completed step", () => {
  const state = createLessonState();
  const next = advanceLesson(state);

  assert.notEqual(next, state);
  assert.equal(next.stepIndex, 1);
  assert.deepEqual(next.completed, ["story"]);
  assert.deepEqual(state.completed, []);
});

test("the prediction step cannot advance before the learner chooses", () => {
  const predictState = advanceLesson(createLessonState());

  assert.throws(
    () => advanceLesson(predictState),
    /Choose a prediction before continuing/,
  );
});

test("recording a prediction unlocks the simulation step", () => {
  const predictState = advanceLesson(createLessonState());
  const answered = recordPrediction(predictState, "gpu");
  const simulateState = advanceLesson(answered);

  assert.equal(answered.prediction, "gpu");
  assert.equal(simulateState.stepIndex, 2);
  assert.deepEqual(simulateState.completed, ["story", "predict"]);
});

test("retreating removes the reopened step from completion history", () => {
  const predictState = advanceLesson(createLessonState());
  const simulateState = advanceLesson(recordPrediction(predictState, "gpu"));
  const returned = retreatLesson(simulateState);

  assert.equal(returned.stepIndex, 1);
  assert.equal(returned.prediction, "gpu");
  assert.deepEqual(returned.completed, ["story"]);
});

test("pixel work maps one active thread to each pixel and masks overflow", () => {
  const work = buildPixelWork({ width: 3, height: 2, blockSize: 4 });

  assert.equal(work.totalPixels, 6);
  assert.equal(work.launchedThreads, 8);
  assert.equal(work.blockCount, 2);
  assert.deepEqual(work.threads[4], {
    threadId: 4,
    blockId: 1,
    laneId: 0,
    pixelIndex: 4,
    x: 1,
    y: 1,
    active: true,
  });
  assert.deepEqual(work.threads[6], {
    threadId: 6,
    blockId: 1,
    laneId: 2,
    pixelIndex: null,
    x: null,
    y: null,
    active: false,
  });
});

test("pixel work rejects dimensions and block sizes that cannot teach clearly", () => {
  assert.throws(
    () => buildPixelWork({ width: 0, height: 2, blockSize: 4 }),
    /width must be a positive integer/,
  );
  assert.throws(
    () => buildPixelWork({ width: 3, height: 2, blockSize: 0 }),
    /blockSize must be a positive integer/,
  );
});

test("grid arrow keys move one cell or one row without leaving the map", () => {
  assert.equal(nextGridIndex({ current: 10, key: "ArrowRight", columns: 8, total: 64 }), 11);
  assert.equal(nextGridIndex({ current: 10, key: "ArrowLeft", columns: 8, total: 64 }), 9);
  assert.equal(nextGridIndex({ current: 10, key: "ArrowDown", columns: 8, total: 64 }), 18);
  assert.equal(nextGridIndex({ current: 10, key: "ArrowUp", columns: 8, total: 64 }), 2);
  assert.equal(nextGridIndex({ current: 0, key: "ArrowLeft", columns: 8, total: 64 }), 0);
  assert.equal(nextGridIndex({ current: 63, key: "ArrowDown", columns: 8, total: 64 }), 63);
  assert.equal(nextGridIndex({ current: 10, key: "Enter", columns: 8, total: 64 }), 10);
});

test("a non-divisible block size exposes masked overflow threads", () => {
  const work = buildPixelWork({ width: 8, height: 8, blockSize: 10 });

  assert.equal(work.totalPixels, 64);
  assert.equal(work.launchedThreads, 70);
  assert.equal(work.threads.filter((thread) => !thread.active).length, 6);
});
