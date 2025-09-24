// worker.js
// Purpose: Securely fetch private JSON from Cloudflare R2 with HMAC-signed requests.
// Access:   GET /?key=<path/in/r2.json>&ts=<unix_seconds>&sig=<hmac_sha256_hex>
// Signature: sig = hex(HMAC_SHA256(HMAC_SECRET, `${key}|${ts}`))
// Security:
//  - HMAC signature verification (shared secret)
//  - Timestamp window (MAX_SKEW_MS) to mitigate replay
//  - Multi-prefix allowlist via ALLOW_PREFIXES (comma-separated)
//  - Optional ETag/Cache-Control and conditional requests

export default {
    async fetch(request, env) {
      try {
        const url = new URL(request.url);
        const key = url.searchParams.get("key") || "";
        const ts  = url.searchParams.get("ts") || "";
        const sig = url.searchParams.get("sig") || "";
  
        // 1) Basic param checks
        if (!key || !ts || !sig) return json({ error: "missing key/ts/sig" }, 400);
        if (key.includes("..")) return json({ error: "invalid key" }, 400);
  
        // 2) Timestamp window (default 5 minutes)
        const maxSkewMs = Number(env.MAX_SKEW_MS || 5 * 60 * 1000);
        const nowMs = Date.now();
        const tsMs = Number(ts) * 1000;
        if (!Number.isFinite(tsMs) || Math.abs(nowMs - tsMs) > maxSkewMs) {
          return json({ error: "stale or invalid ts" }, 401);
        }
  
        // 3) Multi-prefix allowlist (comma-separated)
        //    Example: "cache-debug/,cache/,public/data/"
        const allowPrefixes = (env.ALLOW_PREFIXES || "")
          .split(",")
          .map(p => p.trim())
          .filter(Boolean);
  
        if (allowPrefixes.length > 0 && !allowPrefixes.some(p => key.startsWith(p))) {
          return json({ error: "forbidden key prefix" }, 403);
        }
  
        // 4) HMAC verification: sig = HMAC_SHA256(secret, `${key}|${ts}`)
        const expected = await hmacHex(env.HMAC_SECRET, `${key}|${ts}`);
        if (!timingSafeEqualHex(expected, sig)) {
          return json({ error: "bad signature" }, 401);
        }
  
        // 5) Fetch from R2
        const obj = await env.R2.get(key);
        if (!obj) return json({ error: "not found" }, 404);
  
        // Optional: ETag & Cache-Control
        const etag = obj.httpEtag || `W/"${await sha1Hex(key + (obj.uploaded || ""))}"`;
        const headers = {
          "Content-Type": "application/json; charset=utf-8",
          "Cache-Control": env.CACHE_CONTROL || "private, max-age=60",
          "ETag": etag,
        };
  
        // Conditional requests
        const ifNoneMatch = request.headers.get("If-None-Match");
        if (ifNoneMatch && ifNoneMatch === etag) {
          return new Response(null, { status: 304, headers });
        }
  
        // Return JSON
        const body = await obj.text();
        return new Response(body, { status: 200, headers });
      } catch {
        return json({ error: "internal_error" }, 500);
      }
    },
  };
  
  // ----- utils -----
  function json(data, status = 200, extra = {}) {
    return new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8", ...extra },
    });
  }
  
  async function hmacHex(secret, msg) {
    if (!secret) throw new Error("HMAC_SECRET not set");
    const key = await crypto.subtle.importKey(
      "raw",
      new TextEncoder().encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );
    const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(msg));
    return toHex(new Uint8Array(sig));
  }
  
  async function sha1Hex(text) {
    const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(text));
    return toHex(new Uint8Array(buf));
  }
  
  function toHex(bytes) {
    return [...bytes].map(b => b.toString(16).padStart(2, "0")).join("");
  }
  
  // Constant-time hex compare
  function timingSafeEqualHex(a, b) {
    if (typeof a !== "string" || typeof b !== "string") return false;
    if (a.length !== b.length) return false;
    let out = 0;
    for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
    return out === 0;
  }
  