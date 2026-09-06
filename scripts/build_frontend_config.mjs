import { writeFileSync } from "node:fs";

const apiBase = (process.env.ISHAARA_API_BASE || "").replace(/\/$/, "");

if (apiBase && !/^https:\/\//i.test(apiBase)) {
  throw new Error("ISHAARA_API_BASE must be an HTTPS URL, for example https://ishaara-api.onrender.com");
}

const contents = `document.documentElement.dataset.apiBase = ${JSON.stringify(apiBase)};\n`;
writeFileSync(new URL("../review-demo/config.js", import.meta.url), contents, "utf8");
console.log(`Frontend API target: ${apiBase || "same origin (local development)"}`);
