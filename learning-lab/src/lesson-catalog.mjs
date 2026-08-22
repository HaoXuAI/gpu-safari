export const LESSONS = Object.freeze([
  Object.freeze({ id: "paint-pixels", title: "Paint Pixels in Parallel", summary: "Give every pixel a worker and watch parallel work take shape.", trail: "Foundations", difficulty: "Beginner", durationMinutes: 10, ecosystems: Object.freeze(["Metal", "Triton", "CUDA"]), status: "available", route: "./learn/paint-pixels/" }),
  Object.freeze({ id: "cuda-memory", title: "Move Data with Purpose", summary: "Explore global, shared, and local memory through a visual data relay.", trail: "CUDA", difficulty: "Beginner", durationMinutes: 15, ecosystems: Object.freeze(["CUDA"]), status: "upcoming", route: null }),
  Object.freeze({ id: "triton-tiles", title: "Think in Tiles", summary: "See how Triton programs divide arrays into masked blocks of work.", trail: "Triton", difficulty: "Intermediate", durationMinutes: 15, ecosystems: Object.freeze(["Triton"]), status: "upcoming", route: null }),
  Object.freeze({ id: "rocm-first-kernel", title: "Cross the Vendor Trail", summary: "Carry the same parallel idea onto an AMD GPU with ROCm.", trail: "ROCm", difficulty: "Intermediate", durationMinutes: 20, ecosystems: Object.freeze(["ROCm"]), status: "upcoming", route: null }),
  Object.freeze({ id: "profile-kernels", title: "Follow the Bottleneck", summary: "Read a GPU timeline and discover whether compute, memory, or launch overhead is holding work back.", trail: "Profiling", difficulty: "Intermediate", durationMinutes: 20, ecosystems: Object.freeze(["Nsight", "Metal"]), status: "upcoming", route: null }),
  Object.freeze({ id: "ai-workload", title: "From Kernel to Model", summary: "Connect matrix operations, memory movement, and kernels to a small end-to-end AI workload.", trail: "AI workloads", difficulty: "Intermediate", durationMinutes: 25, ecosystems: Object.freeze(["PyTorch", "MLX"]), status: "upcoming", route: null }),
]);

export function getAvailableLessons() {
  return LESSONS.filter((lesson) => lesson.status === "available");
}

export function getLesson(id) {
  const lesson = LESSONS.find((candidate) => candidate.id === id);
  if (!lesson) throw new Error(`Unknown lesson: ${id}`);
  return lesson;
}
