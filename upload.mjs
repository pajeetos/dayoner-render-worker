// Uploads the finished video to Vercel Blob with the one-file token the job
// brought (BLOB_TOKEN), so this worker never holds the store's own key.
//   node upload.mjs <file> <pathname>  ->  {"url": "...", "bytes": n}
import { createReadStream, statSync } from "node:fs";
import { put } from "@vercel/blob";

const [file, pathname] = process.argv.slice(2);
const token = process.env.BLOB_TOKEN;
if (!file || !pathname || !token) throw new Error("usage: BLOB_TOKEN=... node upload.mjs <file> <pathname>");
const bytes = statSync(file).size;
const blob = await put(pathname, createReadStream(file), {
  access: "public",
  token,
  contentType: "video/mp4",
  multipart: bytes > 50 * 1024 * 1024,
});
console.log(JSON.stringify({ url: blob.url, bytes }));
