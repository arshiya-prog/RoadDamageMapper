// Shared by the dashboard and the phone recorder.
// Uploads a ride in chunks (public tunnels cap a single request at ~100 MB) and retries each
// chunk, because phones on mobile data drop connections.
(function () {
  const CHUNK = 20 * 1024 * 1024;
  const RETRIES = 4;

  // Access key for servers started with ROADSCAN_KEY. Arrives once as ?key=… and is remembered.
  function accessKey() {
    const url = new URL(location.href);
    const fromUrl = url.searchParams.get("key");
    if (fromUrl) {
      try { localStorage.setItem("roadscanKey", fromUrl); } catch {}
      url.searchParams.delete("key");
      history.replaceState(null, "", url);  // keep the key out of the address bar and screenshots
      return fromUrl;
    }
    try { return localStorage.getItem("roadscanKey") || ""; } catch { return ""; }
  }
  const KEY = accessKey();
  const authHeaders = () => (KEY ? { "X-RoadScan-Key": KEY } : {});

  // XMLHttpRequest rather than fetch: only XHR reports how many bytes of a request body have been sent.
  function send(method, path, body, onSent) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(method, path);
      for (const [k, v] of Object.entries(authHeaders())) xhr.setRequestHeader(k, v);
      if (onSent) xhr.upload.onprogress = (e) => onSent(e.loaded);
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch {}
        if (xhr.status === 401) return reject(Object.assign(new Error("Access key missing or wrong — open the link that includes ?key="), { fatal: true }));
        if (xhr.status >= 400 && xhr.status < 500) return reject(Object.assign(new Error(data.detail || xhr.statusText), { fatal: true }));
        if (xhr.status < 200 || xhr.status >= 300) return reject(new Error(data.detail || `HTTP ${xhr.status}`));
        resolve(data);
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.ontimeout = () => reject(new Error("timed out"));
      xhr.send(body ?? null);
    });
  }

  async function request(method, path, body, onSent, attempt = 0) {
    try {
      return await send(method, path, body, onSent);
    } catch (e) {
      if (e.fatal || attempt >= RETRIES) throw e;
      onSent?.(0);  // this piece starts again from zero
      await new Promise((res) => setTimeout(res, 1000 * 2 ** attempt));  // 1, 2, 4, 8 s
      return request(method, path, body, onSent, attempt + 1);
    }
  }

  const mb = (bytes) => (bytes / 1e6).toFixed(1);

  /**
   * video: Blob, videoName: e.g. "ride.mp4", gps: Blob, gpsName: e.g. "track.json",
   * fields: { name, recorded_at, gps_offset_s, gps_source }, onProgress(fraction, text)
   * fraction and text always describe the same moment: bytes of video confirmed sent so far.
   * Resolves to { id, status } for the new ride.
   */
  async function uploadRide({ video, videoName, gps, gpsName, fields = {}, onProgress = () => {} }) {
    const total = video.size;
    const report = (sent) => onProgress(total ? sent / total : 1, `Uploading ${mb(sent)} of ${mb(total)} MB`);
    report(0);
    const { upload_id } = await request("POST", "/api/uploads");
    const parts = Math.max(1, Math.ceil(total / CHUNK));
    for (let i = 0; i < parts; i++) {
      const done = i * CHUNK;  // bytes in pieces already accepted by the server
      const piece = video.slice(done, done + CHUNK);
      await request("PUT", `/api/uploads/${upload_id}/${i}`, piece, (loaded) => report(done + Math.min(loaded, piece.size)));
      report(done + piece.size);
    }
    onProgress(1, "Finishing upload…");
    const fd = new FormData();
    fd.append("parts", String(parts));
    fd.append("filename", videoName);
    fd.append("gps", gps, gpsName);
    for (const [k, v] of Object.entries(fields)) if (v != null && v !== "") fd.append(k, v);
    return request("POST", `/api/uploads/${upload_id}/complete`, fd);
  }

  window.RoadScanUpload = { uploadRide, authHeaders, hasKey: () => !!KEY };
})();
