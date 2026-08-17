export const LESSON_STEPS = Object.freeze([
  "story",
  "predict",
  "simulate",
  "code",
  "explain",
  "challenge",
]);

export function createLessonState() {
  return {
    stepIndex: 0,
    prediction: null,
    completed: [],
  };
}

export function recordPrediction(state, prediction) {
  if (!new Set(["cpu", "gpu", "same"]).has(prediction)) {
    throw new Error("Prediction must be cpu, gpu, or same");
  }

  return { ...state, prediction };
}

export function advanceLesson(state) {
  const currentStep = LESSON_STEPS[state.stepIndex];

  if (currentStep === "predict" && state.prediction === null) {
    throw new Error("Choose a prediction before continuing");
  }

  if (state.stepIndex >= LESSON_STEPS.length - 1) {
    return state;
  }

  return {
    ...state,
    stepIndex: state.stepIndex + 1,
    completed: [...state.completed, currentStep],
  };
}

export function retreatLesson(state) {
  if (state.stepIndex === 0) {
    return state;
  }

  const stepIndex = state.stepIndex - 1;
  return {
    ...state,
    stepIndex,
    completed: LESSON_STEPS.slice(0, stepIndex),
  };
}

function requirePositiveInteger(name, value) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
}

export function buildPixelWork({ width, height, blockSize }) {
  requirePositiveInteger("width", width);
  requirePositiveInteger("height", height);
  requirePositiveInteger("blockSize", blockSize);

  const totalPixels = width * height;
  const blockCount = Math.ceil(totalPixels / blockSize);
  const launchedThreads = blockCount * blockSize;
  const threads = Array.from({ length: launchedThreads }, (_, threadId) => {
    const active = threadId < totalPixels;

    return {
      threadId,
      blockId: Math.floor(threadId / blockSize),
      laneId: threadId % blockSize,
      pixelIndex: active ? threadId : null,
      x: active ? threadId % width : null,
      y: active ? Math.floor(threadId / width) : null,
      active,
    };
  });

  return { totalPixels, launchedThreads, blockCount, threads };
}

export function nextGridIndex({ current, key, columns, total }) {
  const moves = {
    ArrowRight: 1,
    ArrowLeft: -1,
    ArrowDown: columns,
    ArrowUp: -columns,
  };

  if (key === "Home") return 0;
  if (key === "End") return total - 1;
  if (!(key in moves)) return current;

  return Math.min(total - 1, Math.max(0, current + moves[key]));
}
