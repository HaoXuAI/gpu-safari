import test from "node:test";
import assert from "node:assert/strict";

import { getLessonCta, renderLessonCard, renderTrailCard } from "../src/site-view.mjs";

const AVAILABLE = Object.freeze({
  id: "paint-pixels",
  title: "Paint Pixels in Parallel",
  summary: "Give every pixel a worker.",
  trail: "Foundations",
  difficulty: "Beginner",
  durationMinutes: 10,
  ecosystems: ["Metal", "Triton", "CUDA"],
  status: "available",
  route: "./learn/paint-pixels/",
});

test("an available lesson card is an accessible link with useful learning context", () => {
  const html = renderLessonCard(AVAILABLE);

  assert.match(html, /href="\.\/learn\/paint-pixels\/"/);
  assert.match(html, /Paint Pixels in Parallel/);
  assert.match(html, /10 min/);
  assert.match(html, /Beginner/);
  assert.match(html, /Metal · Triton · CUDA/);
});

test("an upcoming trail card has no link and clearly communicates availability", () => {
  const html = renderTrailCard({ ...AVAILABLE, status: "upcoming", route: null }, "../");

  assert.doesNotMatch(html, /href=/);
  assert.match(html, /Coming soon/);
  assert.match(html, /aria-disabled="true"/);
});

test("trail routes resolve relative to the trail map", () => {
  const html = renderTrailCard(AVAILABLE, "../");

  assert.match(html, /href="\.\.\/learn\/paint-pixels\/"/);
});

test("the primary action derives its route and duration from lesson metadata", () => {
  assert.deepEqual(getLessonCta(AVAILABLE), {
    href: "./learn/paint-pixels/",
    label: "Start the 10-minute lesson",
  });
});
