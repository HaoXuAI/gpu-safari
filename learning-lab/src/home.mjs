import { getLesson, LESSONS } from "./lesson-catalog.mjs";
import { getLessonCta, renderLessonCard, renderTrailCard } from "./site-view.mjs";

const featuredLesson = getLesson("paint-pixels");
const primaryAction = document.querySelector("#primary-lesson-link");
const cta = getLessonCta(featuredLesson);
primaryAction.href = cta.href;
primaryAction.replaceChildren(document.createTextNode(`${cta.label} `));
primaryAction.insertAdjacentHTML("beforeend", '<span aria-hidden="true">→</span>');

document.querySelector("#featured-lesson").innerHTML = renderLessonCard(featuredLesson);
document.querySelector("#trail-preview").innerHTML = LESSONS.slice(1).map((lesson) => renderTrailCard(lesson)).join("");
