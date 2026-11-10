// Uploads a file to Vercel Blob with the one-file token the job brought (BLOB_TOKEN), so this worker never
// holds the store's own key: the finished video, or the files two GPUs share for one scene.
//   node upload.mjs <file> <pathname> [content type, video/mp4]  ->  {"url": "...", "bytes": n}
import { createReadStream, statSync } from "node:fs";
import { put } from "@vercel/blob";

const [file, pathname, contentType = "video/mp4"] = process.argv.slice(2);
const token = process.env.BLOB_TOKEN;
if (!file || !pathname || !token) throw new Error("usage: BLOB_TOKEN=... node upload.mjs <file> <pathname> [content type]");
const bytes = statSync(file).size;
const blob = await put(pathname, createReadStream(file), {
  access: "public",
  token,
  contentType,
  multipart: bytes > 50 * 1024 * 1024,
});
console.log(JSON.stringify({ url: blob.url, bytes }));
