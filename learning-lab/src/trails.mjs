import { LESSONS } from "./lesson-catalog.mjs";
import { renderTrailCard } from "./site-view.mjs";

document.querySelector("#trail-list").innerHTML = LESSONS.map((lesson) => renderTrailCard(lesson, "../")).join("");
