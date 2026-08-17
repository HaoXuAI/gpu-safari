const LESSON_COPY = Object.freeze({
  story: {
    eyebrow: "Meet the workers",
    title: "Paint pixels in parallel",
  },
  predict: {
    eyebrow: "Make a prediction",
    title: "Which team finishes the picture first?",
  },
  simulate: {
    eyebrow: "Concept simulation",
    title: "Watch threads find their pixels",
  },
  code: {
    eyebrow: "Reveal the code",
    title: "The same idea, three levels closer to the GPU",
  },
  run: {
    eyebrow: "Measured execution",
    title: "Run the idea on a real GPU",
  },
  explain: {
    eyebrow: "Build the mental model",
    title: "Every worker needs an identity and a destination",
  },
  challenge: {
    eyebrow: "Try a change",
    title: "What happens when the team size changes?",
  },
});

const PREDICTION_FEEDBACK = Object.freeze({
  gpu: "That is the parallel idea: many GPU workers can each own one pixel. Now let’s inspect how the work is assigned.",
  cpu: "That is a reasonable prediction—a CPU worker is individually powerful. The simulation will show why many independent pixels are a useful GPU-shaped problem.",
  same: "That is a reasonable baseline. The amount of work is identical; the important question is how much of it can happen in parallel.",
});

export function getLessonCopy(step) {
  const copy = LESSON_COPY[step];
  if (!copy) {
    throw new Error(`Unknown lesson step: ${step}`);
  }
  return copy;
}

export function getPredictionFeedback(prediction) {
  const feedback = PREDICTION_FEEDBACK[prediction];
  if (!feedback) {
    throw new Error(`Unknown prediction: ${prediction}`);
  }
  return feedback;
}

export const CODE_SAMPLES = Object.freeze({
  python: `for y in range(height):\n    for x in range(width):\n        image[y, x] = color`,
  pytorch: `image = torch.empty((height, width, 3), device="cuda")\nimage[:] = color`,
  triton: `@triton.jit\ndef paint(image, color, n_pixels: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n    program = tl.program_id(0)\n    offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_pixels\n    tl.store(image + offsets, color, mask=mask)`,
  cuda: `int pixel = blockIdx.x * blockDim.x + threadIdx.x;\nif (pixel < n_pixels) {\n    image[pixel] = color;\n}`,
});
