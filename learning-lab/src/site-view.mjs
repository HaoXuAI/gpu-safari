function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function lessonMeta(lesson) {
  return `${lesson.durationMinutes} min · ${lesson.difficulty}`;
}

function resolveRoute(route, prefix) {
  return `${prefix}${route.replace(/^\.\//, "")}`;
}

export function getLessonCta(lesson) {
  return {
    href: lesson.route,
    label: `Start the ${lesson.durationMinutes}-minute lesson`,
  };
}

export function renderLessonCard(lesson, prefix = "./") {
  const route = resolveRoute(lesson.route, prefix);
  return `<a class="lesson-card" href="${escapeHtml(route)}">
    <span class="card-kicker">${escapeHtml(lesson.trail)} · Available now</span>
    <strong>${escapeHtml(lesson.title)}</strong>
    <p>${escapeHtml(lesson.summary)}</p>
    <span class="card-meta">${escapeHtml(lessonMeta(lesson))}</span>
    <span class="ecosystems">${lesson.ecosystems.map(escapeHtml).join(" · ")}</span>
    <span class="card-action">Begin expedition <span aria-hidden="true">→</span></span>
  </a>`;
}

export function renderTrailCard(lesson, prefix = "./") {
  const content = `<span class="card-kicker">${escapeHtml(lesson.trail)}</span>
    <strong>${escapeHtml(lesson.title)}</strong>
    <p>${escapeHtml(lesson.summary)}</p>
    <span class="card-meta">${escapeHtml(lessonMeta(lesson))} · ${lesson.ecosystems.map(escapeHtml).join(" · ")}</span>`;

  if (lesson.status === "available") {
    return `<a class="trail-card trail-card-available" href="${escapeHtml(resolveRoute(lesson.route, prefix))}">${content}<span class="card-action">Start lesson <span aria-hidden="true">→</span></span></a>`;
  }
  return `<article class="trail-card trail-card-upcoming" aria-disabled="true">${content}<span class="availability">Coming soon</span></article>`;
}
