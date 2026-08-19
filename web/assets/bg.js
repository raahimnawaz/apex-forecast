/* Mounts the page-wide silk field.
 *
 * Its own entry point rather than part of hero.js, because method.html has a background
 * but no hero — and because the field is the one thing both pages share.
 *
 * The version query is repeated on the import for the same reason it appears in the HTML:
 * a module's imports are fetched as separate requests with their own cache entries, so
 * `bg.js?v=N` alone would leave a rewritten silk.js served from cache.
 */
import { mountSilkBackground } from "./silk.js?v=24";

const el = document.getElementById("silk-bg");
if (el) mountSilkBackground(el, { seed: 41 });
