// ---------- helpers ----------
const COLORS = { critical: "#b42318", high: "#dd6b20", medium: "#d69e2e", low: "#2f855a", repaired: "#8a929c" };
const LEVELS = ["critical", "high", "medium", "low"];
const LEVEL_RANK = { critical: 0, high: 1, medium: 2, low: 3, repaired: 4 };
const GPS_TEXT = {
  device: "Phone GPS",
  reconstructed: "GPS reconstructed (±10–20 m)",
  simulated: "Simulated GPS",
};
const SYNC_TEXT = {
  device: "Map position comes from the phone's GPS log, matched to the video clock.",
  reconstructed: "GPS was reconstructed from a map route and video landmarks after the ride — map positions are approximate (±10–20 m).",
  simulated: "Synthetic ride: simulated GPS and generated detections, no video.",
};
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(r.statusText); return r.json(); });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtDate = (iso) => new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
const fmtShort = (iso) => new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
const pct = (c) => `${Math.round(c * 100)}%`;
const sevKey = (d) => (d.status === "possibly_repaired" ? "repaired" : d.level);
const sourceTag = (src) => (src === "synthetic"
  ? '<span class="tag synthetic" title="Generated demo data — not detected">Synthetic</span>'
  : '<span class="tag model" title="Detected by the trained model">Model prediction</span>');
const levelBadge = (d) => `<span class="badge sev-${sevKey(d)}">${d.status === "possibly_repaired" ? "Possibly repaired" : cap(d.level)}</span>`;
// Floor, not round: levels use "< 75" style cut-offs, so 74.6 must read 74 (High), not 75.
const sevNum = (v) => Math.floor(v);
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const withSource = (path) => (state.source === "all" ? path : `${path}${path.includes("?") ? "&" : "?"}source=${state.source}`);

function pinIcon(level, { synthetic = false, selected = false, current = false, pulse = false } = {}) {
  const cls = ["pin", `sev-${level}`, level === "critical" ? "big" : "", synthetic ? "synthetic" : "",
    selected ? "selected" : "", current ? "current" : "", pulse ? "pulse" : ""].join(" ");
  const size = level === "critical" ? 20 : 16;
  return L.divIcon({ className: "", html: `<div class="${cls}"></div>`, iconSize: [size, size] });
}

// ---------- state ----------
const DEFAULT_FILTERS = () => ({ levels: new Set(LEVELS), status: "active", trend: null, multi: false });
const state = {
  source: "all",
  filters: DEFAULT_FILTERS(),
  preset: "active",
  stats: null,
  rides: [],
  defects: [],
  selectedId: null,
  ride: null,
  t: 0,
  playing: false,
  shown: new Map(),
  live: null,
  tab: "replay",
};

// ---------- map ----------
const map = L.map("map", { zoomControl: true, maxZoom: 19 }).setView([18.545, 73.80], 13);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', maxZoom: 19,
  className: "basemap",
}).addTo(map);
const rideLayer = L.layerGroup().addTo(map);
const defectLayer = L.markerClusterGroup({
  maxClusterRadius: 40,
  showCoverageOnHover: false,
  disableClusteringAtZoom: 18,
  spiderfyOnMaxZoom: true,
  // Cluster takes the colour of the worst defect inside it.
  iconCreateFunction(cluster) {
    const worst = cluster.getAllChildMarkers().map((m) => m.options.sev).sort((a, b) => LEVEL_RANK[a] - LEVEL_RANK[b])[0];
    const n = cluster.getChildCount();
    const size = n < 10 ? 32 : n < 30 ? 38 : 44;
    return L.divIcon({ className: "", html: `<div class="cluster sev-${worst}" aria-label="${n} defects">${n}</div>`, iconSize: [size, size] });
  },
});

// ---------- header, banner & stats ----------
$$(".seg button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.source)));

async function setSource(src) {
  state.source = src;
  $$(".seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.source === src)));
  $("#csv-link").href = withSource("/api/priority.csv");
  closeDetail();
  await Promise.all([loadStats(), loadDefects(), loadRides(undefined, { keepCurrent: true })]);
}

async function loadStats() {
  const [s, all] = await Promise.all([api(withSource("/api/stats")), state.source === "all" ? null : api("/api/stats")]);
  state.stats = s;
  const bySrc = (all || s).by_source;
  $("#count-model").textContent = `(${bySrc.model.rides})`;
  $("#count-synthetic").textContent = `(${bySrc.synthetic.rides})`;
  renderScopeLine();
  renderBanner(bySrc);

  const cards = [
    { key: "rides", value: `${s.rides}`, label: `rides · ${s.km_surveyed} km`, title: "Open ride replay" },
    { key: "active", value: s.active_defects, label: "active defects", title: "Show all active defects" },
    { key: "critical", value: s.critical, label: "critical", alert: true, title: "Show critical defects in the worklist" },
    { key: "worsening", value: s.worsening, label: "worsening", alert: true, title: "Show defects whose severity rose ≥10 pts over ≥7 days" },
    { key: "multi", value: s.multi_ride, label: "confirmed on 2+ rides", title: "Show defects detected on more than one ride" },
    { key: "repaired", value: s.possibly_repaired, label: "possibly repaired", title: "Show defects missed on 2+ later passes" },
  ];
  $("#stats").innerHTML = cards.map((c) => `<button class="stat ${c.alert && +c.value ? "alert" : ""}" data-preset="${c.key}"
      aria-pressed="${state.preset === c.key}" title="${esc(c.title)}"><b>${c.value}</b><small>${esc(c.label)}</small></button>`).join("");
  $$("#stats .stat").forEach((b) => b.addEventListener("click", () => applyPreset(b.dataset.preset)));
}

function renderScopeLine() {
  const s = state.stats;
  if (!s) return;
  const scope = { all: "All data", model: "Model predictions only", synthetic: "Synthetic demo only" }[state.source];
  const range = s.date_from ? (s.date_from.slice(0, 10) === s.date_to.slice(0, 10)
    ? fmtDate(s.date_from) : `${fmtDate(s.date_from)} – ${fmtDate(s.date_to)}`) : "no rides yet";
  const proc = s.processing ? `${s.processing} ride(s) processing` : s.failed ? `${s.failed} ride(s) failed` : "all rides processed";
  const viewing = state.tab === "replay" && state.ride
    ? `Replaying: ${state.ride.name}`
    : { defects: "Viewing: defect map", worklist: "Viewing: repair worklist" }[state.tab] || "";
  $("#scope-line").textContent = [scope, `${s.rides} ride(s), ${range}`, proc, viewing].filter(Boolean).join(" · ");
}

function renderBanner(bySrc) {
  const b = $("#data-banner");
  if (state.source === "synthetic") {
    b.innerHTML = "<b>Synthetic demo data.</b> Every ride, defect, image, confidence and severity shown here was generated by the demo script — nothing was detected by the model.";
  } else if (state.source === "all" && bySrc.synthetic.rides) {
    b.innerHTML = `<b>Mixed data.</b> ${bySrc.synthetic.rides} synthetic demo ride(s) are shown alongside ${bySrc.model.rides} ride(s) processed by the model. Synthetic items are tagged <span class="tag synthetic">Synthetic</span> and drawn as rings on the map. <a href="#" data-source-link="model">Show model predictions only</a>`;
    $("[data-source-link]", b).addEventListener("click", (e) => { e.preventDefault(); setSource("model"); });
  } else {
    b.innerHTML = "";
  }
}

// ---------- filters & presets ----------
const PRESETS = {
  active: { tab: "defects", f: () => DEFAULT_FILTERS() },
  critical: { tab: "worklist", f: () => ({ ...DEFAULT_FILTERS(), levels: new Set(["critical"]) }) },
  worsening: { tab: "worklist", f: () => ({ ...DEFAULT_FILTERS(), trend: "worsening" }) },
  multi: { tab: "defects", f: () => ({ ...DEFAULT_FILTERS(), multi: true }) },
  repaired: { tab: "defects", f: () => ({ ...DEFAULT_FILTERS(), status: "repaired" }) },
};

function applyPreset(key) {
  if (key === "rides") { state.preset = null; switchTab("replay"); refreshPresetButtons(); return; }
  state.filters = PRESETS[key].f();
  state.preset = key;
  switchTab(PRESETS[key].tab);
  refreshPresetButtons();
}

function refreshPresetButtons() {
  $$("#stats .stat").forEach((b) => b.setAttribute("aria-pressed", String(state.preset === b.dataset.preset)));
}

function filteredDefects() {
  const f = state.filters;
  return state.defects.filter((d) =>
    f.levels.has(d.level)
    && (f.status === "all" || (f.status === "active" ? d.status === "active" : d.status === "possibly_repaired"))
    && (!f.trend || d.trend_status === f.trend)
    && (!f.multi || d.n_obs >= 2));
}

function renderFilters(container) {
  const f = state.filters;
  const chip = (attr, label, pressed) => `<button class="chip" ${attr} aria-pressed="${pressed}">${label}</button>`;
  const n = filteredDefects().length;
  container.innerHTML = `
    ${LEVELS.map((l) => chip(`data-level="${l}"`, cap(l), f.levels.has(l))).join("")}
    <span class="chip-sep"></span>
    ${chip('data-status="active"', "Active", f.status === "active")}
    ${chip('data-status="repaired"', "Possibly repaired", f.status === "repaired")}
    ${chip('data-status="all"', "Any status", f.status === "all")}
    <span class="chip-sep"></span>
    ${chip('data-trend="worsening"', "Worsening", f.trend === "worsening")}
    ${chip('data-multi="1"', "2+ rides", f.multi)}
    <div class="filter-summary"><span>${n} of ${state.defects.length} defects match</span>
      <button class="linkbtn" data-reset>Reset filters</button></div>`;
  $$("[data-level]", container).forEach((b) => b.addEventListener("click", () => {
    const l = b.dataset.level;
    if (f.levels.has(l) && f.levels.size > 1) f.levels.delete(l); else f.levels.add(l);
    filtersChanged();
  }));
  $$("[data-status]", container).forEach((b) => b.addEventListener("click", () => { f.status = b.dataset.status; filtersChanged(); }));
  $("[data-trend]", container).addEventListener("click", () => { f.trend = f.trend ? null : "worsening"; filtersChanged(); });
  $("[data-multi]", container).addEventListener("click", () => { f.multi = !f.multi; filtersChanged(); });
  $("[data-reset]", container).addEventListener("click", () => { state.filters = DEFAULT_FILTERS(); filtersChanged(); });
}

function filtersChanged() {
  state.preset = Object.keys(PRESETS).find((k) => sameFilters(PRESETS[k].f(), state.filters)) || null;
  refreshPresetButtons();
  renderDefects();
  renderWorklist();
  fitToDefects();
}

function sameFilters(a, b) {
  return a.status === b.status && a.trend === b.trend && a.multi === b.multi
    && a.levels.size === b.levels.size && [...a.levels].every((l) => b.levels.has(l));
}

// ---------- tabs ----------
$$(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

function switchTab(tab) {
  state.tab = tab;
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
  if (tab === "replay") {
    map.removeLayer(defectLayer);
    rideLayer.addTo(map);
    if (state.ride?._route) map.fitBounds(state.ride._route.getBounds().pad(0.08));
  } else {
    pause();
    map.removeLayer(rideLayer);
    defectLayer.addTo(map);
    renderDefects();
    renderWorklist();
    fitToDefects();
  }
  renderScopeLine();
}

function fitToDefects() {
  const list = filteredDefects();
  if (state.tab === "replay" || !list.length) return;
  map.fitBounds(L.latLngBounds(list.map((d) => [d.lat, d.lon])).pad(0.15), { maxZoom: 17 });
}

// ---------- rides & replay ----------
async function loadRides(selectId, { keepCurrent = false } = {}) {
  state.rides = await api("/api/rides");
  const visible = state.rides.filter((r) => state.source === "all" || r.source === state.source);
  const sel = $("#ride-select");
  const opt = (r) => `<option value="${r.id}">${esc(r.name)} · ${fmtShort(r.recorded_at)}${r.status !== "done" ? ` (${r.status})` : ""}</option>`;
  const groups = [["model", "Model predictions"], ["synthetic", "Synthetic demo"]]
    .map(([src, label]) => [label, visible.filter((r) => r.source === src)])
    .filter(([, rs]) => rs.length);
  sel.innerHTML = groups.map(([label, rs]) => `<optgroup label="${label}">${rs.map(opt).join("")}</optgroup>`).join("");
  if (!visible.length) {
    sel.innerHTML = "<option>No rides in this view</option>";
    state.ride = null;
    rideLayer.clearLayers();
    $("#ride-meta").innerHTML = "";
    $("#feed").innerHTML = "";
    $("#feed-count").textContent = "0";
    $("#ticks").innerHTML = "";
    return;
  }
  let id = selectId;
  if (id == null && keepCurrent && visible.some((r) => r.id === state.ride?.id)) id = state.ride.id;
  // Default: most recent model ride if there is one (real output first), else most recent ride.
  if (id == null) id = (groups[0][1].at(-1) || visible.at(-1)).id;
  sel.value = id;
  if (state.ride?.id !== id || !keepCurrent) await openRide(id);
}
$("#ride-select").addEventListener("change", (e) => openRide(+e.target.value));

async function openRide(id) {
  pause();
  if (state.live) { state.live.close(); state.live = null; }
  const ride = await api(`/api/rides/${id}`);
  state.ride = ride;
  state.t = 0;
  state.shown.clear();
  rideLayer.clearLayers();
  $("#feed").innerHTML = "";
  $("#feed-count").textContent = "0";
  $("#ride-select").value = id;

  const latlngs = ride.track.map(([, lat, lon]) => [lat, lon]);
  ride._route = L.polyline(latlngs, { color: "#545b66", weight: 4, opacity: 0.6, dashArray: "2 8", lineCap: "round" }).addTo(rideLayer);
  ride._done = L.polyline([], { color: "#2457c5", weight: 5, opacity: 0.8 }).addTo(rideLayer);
  ride._vehicle = L.marker(latlngs[0], {
    icon: L.divIcon({ className: "", html: '<div class="vehicle" title="Vehicle position"></div>', iconSize: [18, 18] }),
    zIndexOffset: 1000, keyboard: false,
  }).addTo(rideLayer);
  map.invalidateSize();  // the map may have been laid out before its container had a size
  if (state.tab === "replay") map.fitBounds(ride._route.getBounds().pad(0.08));

  $("#ride-meta").innerHTML = `${sourceTag(ride.source)}
    <span class="tag gps" title="${esc(SYNC_TEXT[ride.gps_source])}">${GPS_TEXT[ride.gps_source] || ride.gps_source}</span>
    ${ride.status !== "done" ? `<span class="tag status">${esc(ride.status)}</span>` : ""}
    <span>${fmtDate(ride.recorded_at)} · ${(ride.distance_m / 1000).toFixed(1)} km · ${fmtTime(ride.duration_s)} · ${ride.n_detections} detection(s)</span>
    <button class="btn small danger" id="delete-ride-btn" ${ride.status === "pending" || ride.status === "processing"
      ? 'disabled title="Wait for processing to finish before deleting"' : 'title="Delete this ride"'}>Delete ride</button>`;
  $("#delete-ride-btn").addEventListener("click", () => openDeleteDialog(ride));
  $("#sync-note").textContent = SYNC_TEXT[ride.gps_source] || "";

  const video = $("#video");
  $(".dashcam").classList.toggle("has-video", !!ride.video_url);
  if (ride.video_url) { video.src = ride.video_url; video.load(); } else { video.removeAttribute("src"); video.load(); }
  renderTicks();
  update(0);
  renderScopeLine();

  if (ride.status === "pending" || ride.status === "processing") watchLive(ride.id);
  else $("#live-banner").classList.add("hidden");
  if (ride.status === "failed") showLiveBanner(`Processing failed: ${esc(ride.error)}`, null);
}

function duration() { return state.ride ? Math.max(state.ride.duration_s, 1) : 1; }

function renderTicks() {
  $("#ticks").innerHTML = state.ride.detections.map((d) =>
    `<i style="left:${(d.t / duration()) * 100}%;background:${COLORS[d.level]}" title="${esc(d.label)} at ${fmtTime(d.t)}"></i>`).join("");
}

function positionAt(t) {
  const tr = state.ride.track;
  if (t <= tr[0][0]) return [tr[0][1], tr[0][2], 0];
  for (let i = 1; i < tr.length; i++) {
    if (tr[i][0] >= t) {
      const [t0, a0, o0] = tr[i - 1], [t1, a1, o1] = tr[i];
      const k = (t - t0) / (t1 - t0 || 1);
      return [a0 + (a1 - a0) * k, o0 + (o1 - o0) * k, i];
    }
  }
  const last = tr[tr.length - 1];
  return [last[1], last[2], tr.length];
}

function speedAt(t) {
  const [a, b] = [positionAt(t - 2), positionAt(t + 2)];
  return Math.round((map.distance([a[0], a[1]], [b[0], b[1]]) / 4) * 3.6);
}

// Single place where replay time changes: vehicle, trail, clock, scrubber, markers and feed all follow it.
function update(t) {
  const ride = state.ride;
  if (!ride) return;
  state.t = Math.max(0, Math.min(t, duration()));
  const [lat, lon, idx] = positionAt(state.t);
  ride._vehicle.setLatLng([lat, lon]);
  ride._done.setLatLngs([...ride.track.slice(0, idx).map(([, a, o]) => [a, o]), [lat, lon]]);
  if (state.playing && state.tab === "replay" && !map.getBounds().pad(-0.3).contains([lat, lon])) map.panTo([lat, lon]);

  $("#clock").textContent = `${fmtTime(state.t)} / ${fmtTime(duration())}`;
  $("#scrub").value = Math.round((state.t / duration()) * 1000);
  $("#hud-speed").textContent = speedAt(state.t);

  for (const d of ride.detections) {
    const visible = d.t <= state.t;
    if (visible && !state.shown.has(d.id)) showDetection(d, state.playing);
    else if (!visible && state.shown.has(d.id)) hideDetection(d);
    const s = state.shown.get(d.id);
    if (s) {
      const current = Math.abs(d.t - state.t) <= 2;
      if (s.current !== current) {
        s.current = current;
        s.li.classList.toggle("current", current);
        s.marker.setIcon(detectionIcon(d, { current }));
      }
    }
  }
}

function detectionIcon(d, extra = {}) {
  return pinIcon(d.level, { synthetic: state.ride.source === "synthetic", selected: d.defect_id === state.selectedId, ...extra });
}

function showDetection(d, animate) {
  const m = L.marker([d.lat, d.lon], { icon: detectionIcon(d, { pulse: animate }), title: `${d.label} at ${fmtTime(d.t)}` })
    .addTo(rideLayer).on("click", () => openDefect(d.defect_id));
  const li = document.createElement("li");
  li.className = animate ? "new" : "";
  li.tabIndex = 0;
  li.innerHTML = `${d.crop_url ? `<img class="thumb" src="${d.crop_url}" alt="">` : "<span></span>"}
    <div><div class="title">${esc(d.label)}</div>
    <div class="sub">at ${fmtTime(d.t)} · confidence ${pct(d.conf)} · severity index ${sevNum(d.severity)}</div></div>
    ${levelBadge(d)}`;
  const go = () => { seek(d.t); openDefect(d.defect_id, { fly: false }); };
  li.addEventListener("click", go);
  li.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  $("#feed").prepend(li);
  state.shown.set(d.id, { marker: m, li, current: false });
  $("#feed-count").textContent = state.shown.size;

  if (animate && (d.level === "critical" || d.level === "high")) {
    const f = $("#flash");
    f.style.borderColor = COLORS[d.level];
    f.classList.add("on");
    setTimeout(() => f.classList.remove("on"), 350);
  }
}

function hideDetection(d) {
  const s = state.shown.get(d.id);
  rideLayer.removeLayer(s.marker);
  s.li.remove();
  state.shown.delete(d.id);
  $("#feed-count").textContent = state.shown.size;
}

// ---------- playback ----------
// With a video, the video element is the clock (map follows video.currentTime). Without one, a timer drives replay.
let lastFrame = 0;
function tick(now) {
  if (!state.playing) return;
  const video = $("#video");
  if (state.ride.video_url) {
    update(video.currentTime);
    if (video.ended) { pause(); return; }
  } else {
    update(state.t + ((now - lastFrame) / 1000) * +$("#speed").value);
    if (state.t >= duration()) { pause(); return; }
  }
  lastFrame = now;
  requestAnimationFrame(tick);
}

function play() {
  if (!state.ride) return;
  if (state.t >= duration() - 0.1) seek(0);
  state.playing = true;
  const [lat, lon] = positionAt(state.t);
  if (map.getZoom() < 16) map.setView([lat, lon], 16);
  $("#play").textContent = "❚❚ Pause";
  $(".dashcam").classList.add("playing");
  const video = $("#video");
  if (state.ride.video_url) {
    if (Math.abs(video.currentTime - state.t) > 0.3) video.currentTime = state.t;
    video.playbackRate = Math.min(+$("#speed").value, 16);
    video.play().catch(() => {});
  }
  lastFrame = performance.now();
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  $("#play").textContent = "▶ Play";
  $(".dashcam").classList.remove("playing");
  $("#video").pause();
}

function seek(t) {
  if (state.ride?.video_url) $("#video").currentTime = t;
  update(t);
}

$("#play").addEventListener("click", () => (state.playing ? pause() : play()));
$("#speed").addEventListener("change", () => {
  const v = +$("#speed").value;
  $("#video").playbackRate = Math.min(v, 16);
  $("#sync-note").textContent = (state.ride?.video_url && v > 16 ? "Browsers cap video at 16×; the map follows the video. " : "")
    + (SYNC_TEXT[state.ride?.gps_source] || "");
});
$("#scrub").addEventListener("input", (e) => seek((e.target.value / 1000) * duration()));
$("#video").addEventListener("seeked", () => { if (!state.playing) update($("#video").currentTime); });
// rAF is paused when the page is in the background; timeupdate (~4 Hz) keeps the map on the video clock regardless.
$("#video").addEventListener("timeupdate", () => { if (state.playing && state.ride?.video_url) update($("#video").currentTime); });
$("#video").addEventListener("ended", () => { if (state.playing) pause(); });

// ---------- live processing ----------
function showLiveBanner(text, progress) {
  const b = $("#live-banner");
  b.classList.remove("hidden");
  b.innerHTML = `${text}${progress != null ? `<div class="bar"><i style="width:${progress * 100}%"></i></div>` : ""}`;
}

function watchLive(rideId) {
  showLiveBanner("Waiting to process…", 0);
  const es = new EventSource(`/api/rides/${rideId}/live`);
  state.live = es;
  es.addEventListener("detection", (e) => {
    const d = JSON.parse(e.data);
    if (state.ride?.id !== rideId || state.ride.detections.some((x) => x.id === d.id)) return;
    state.ride.detections.push(d);
    renderTicks();
    showDetection(d, true);
    if (state.tab === "replay") map.panTo([d.lat, d.lon]);
  });
  es.addEventListener("progress", async (e) => {
    const p = JSON.parse(e.data);
    if (p.status === "processing") showLiveBanner(`Detecting road damage… ${Math.round(p.progress * 100)}%`, p.progress);
    if (p.status === "done" || p.status === "failed") {
      es.close();
      state.live = null;
      if (p.status === "failed") showLiveBanner(`Processing failed: ${esc(p.error)}`, null);
      else $("#live-banner").classList.add("hidden");
      await Promise.all([loadStats(), loadDefects()]);
      if (p.status === "done") await loadRides(rideId);
    }
  });
}

// ---------- upload ----------
const dlg = $("#upload-dialog");
$("#upload-btn").addEventListener("click", () => dlg.showModal());
$("#upload-cancel").addEventListener("click", () => dlg.close());
$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const video = fd.get("video"), gps = fd.get("gps");
  const ra = fd.get("recorded_at");
  const submit = form.querySelector('[type="submit"]');
  submit.disabled = true;
  try {
    const body = await RoadScanUpload.uploadRide({
      video, videoName: video.name, gps, gpsName: gps.name,
      fields: { name: fd.get("name"), recorded_at: ra ? new Date(ra).toISOString() : "",
        gps_offset_s: fd.get("gps_offset_s"), gps_source: fd.get("gps_source") },
      onProgress: (f, text) => { $("#upload-status").textContent = `${text} (${Math.floor(f * 100)}%)`; },
    });
    dlg.close();
    form.reset();
    $("#upload-status").textContent = "";
    if (state.source === "synthetic") await setSource("all");
    switchTab("replay");
    await loadRides(body.id);
  } catch (err) {
    $("#upload-status").textContent = `Upload failed: ${err.message}`;
  } finally {
    submit.disabled = false;
  }
});

// ---------- delete ride ----------
const delDlg = $("#delete-dialog");
let deleting = null;

function openDeleteDialog(ride) {
  deleting = ride;
  const n = (k, one, many) => `${k} ${k === 1 ? one : many}`;
  $("#delete-name").textContent = `${ride.name} · ${fmtDate(ride.recorded_at)}`;
  $("#delete-impact").innerHTML = [
    n(ride.n_detections, "detection", "detections") + " from this ride",
    n(ride.defects_only_here, "defect that was only seen on this ride", "defects that were only seen on this ride"),
    ride.video_url ? "The ride's video and detection images" : "The ride's GPS track",
  ].map((t) => `<li>${esc(t)}</li>`).join("");
  $("#delete-error").textContent = "";
  $("#delete-confirm").disabled = false;
  delDlg.showModal();
  $("#delete-cancel").focus();  // safe default: Enter cancels
}

$("#delete-cancel").addEventListener("click", () => delDlg.close());
$("#delete-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ride = deleting;
  $("#delete-confirm").disabled = true;
  $("#delete-error").textContent = "";
  try {
    await RoadScanUpload.deleteRide(ride.id);
  } catch (err) {
    $("#delete-error").textContent = `Couldn't delete: ${err.message}`;
    $("#delete-confirm").disabled = false;
    return;
  }
  delDlg.close();
  pause();
  if (state.live) { state.live.close(); state.live = null; }
  state.ride = null;
  rideLayer.clearLayers();
  await Promise.all([loadStats(), loadDefects()]);
  if (state.selectedId && !state.defects.some((d) => d.id === state.selectedId)) closeDetail();
  await loadRides();
  showToast(`Deleted “${ride.name}”.`);
});

function showToast(text) {
  const t = document.createElement("div");
  t.className = "toast";
  t.setAttribute("role", "status");
  t.textContent = text;
  document.body.append(t);
  setTimeout(() => t.remove(), 4000);
}

// ---------- defects ----------
async function loadDefects() {
  state.defects = await api(withSource("/api/defects"));
  for (const d of state.defects) {
    d._marker = L.marker([d.lat, d.lon], { icon: defectIcon(d), sev: sevKey(d), title: `${d.code} ${d.label}` })
      .on("click", () => openDefect(d.id, { fly: false }));
    d._marker.bindTooltip(`${d.code} · ${d.label} · ${d.status === "active" ? cap(d.level) : "possibly repaired"}`);
  }
  renderDefects();
  renderWorklist();
}

function defectIcon(d) {
  return pinIcon(sevKey(d), { synthetic: d.source === "synthetic", selected: d.id === state.selectedId });
}

function trendShort(d) {
  return { single: "1 ride", too_close: `${d.n_obs} rides, too close to judge`, worsening: "▲ worsening",
    decreased: "▼ fell", stable: "no significant change" }[d.trend_status];
}

function renderDefects() {
  renderFilters($("#defect-filters"));
  const list = filteredDefects();
  defectLayer.clearLayers();
  defectLayer.addLayers(list.map((d) => d._marker));
  const ul = $("#defect-list");
  ul.innerHTML = list.length ? "" : '<li class="empty">No defects match these filters.</li>';
  for (const d of list) {
    const li = document.createElement("li");
    li.tabIndex = 0;
    li.dataset.id = d.id;
    li.className = d.id === state.selectedId ? "selected" : "";
    li.innerHTML = `<span class="mono">${d.code}</span>
      <div><div class="title">${esc(d.label)} ${state.source === "all" ? sourceTag(d.source) : ""}</div>
      <div class="sub">Priority ${d.priority.toFixed(0)} · ${trendShort(d)} · last seen ${fmtShort(d.last_seen)}</div></div>
      ${levelBadge(d)}`;
    li.addEventListener("click", () => openDefect(d.id));
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") openDefect(d.id); });
    ul.append(li);
  }
}

function renderWorklist() {
  renderFilters($("#worklist-filters"));
  const list = filteredDefects();
  const ol = $("#worklist");
  ol.innerHTML = list.length ? "" : '<li class="empty">No defects match these filters.</li>';
  for (const d of list) {
    const li = document.createElement("li");
    li.tabIndex = 0;
    li.dataset.id = d.id;
    li.className = [d.status === "active" ? "" : "repaired", d.id === state.selectedId ? "selected" : ""].join(" ");
    li.innerHTML = `
      <div class="wl-head">
        <div class="wl-rank" title="Rank in this view">${d.rank}</div>
        <div>
          <div class="wl-title">${d.code} · ${esc(d.label)}</div>
          <div class="wl-tags">${levelBadge(d)} ${sourceTag(d.source)}</div>
        </div>
        <div class="wl-score" title="Priority score — see 'How scores work'"><b>${d.priority.toFixed(0)}</b><small>priority / 100</small></div>
      </div>
      <div class="wl-bar" aria-hidden="true"><i style="width:${d.priority}%"></i></div>
      <dl class="wl-grid">
        <dt>Severity index</dt><dd>${sevNum(d.severity)} / 100 (${d.level})</dd>
        <dt>Location</dt><dd><span class="mono">${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}</span>${d.gps_sources.includes("reconstructed") ? " · approx." : ""}</dd>
        <dt>Last observed</dt><dd>${fmtDate(d.last_seen)}${d.last_ride ? ` · ${esc(d.last_ride)}` : ""}</dd>
        <dt>Confirmations</dt><dd>${d.n_obs} independent ride${d.n_obs === 1 ? "" : "s"}</dd>
        <dt>Change</dt><dd>${esc(d.trend_label)}</dd>
      </dl>
      <div class="wl-action">→ ${esc(d.action)}</div>`;
    li.addEventListener("click", () => openDefect(d.id));
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") openDefect(d.id); });
    ol.append(li);
  }
}

// ---------- selected defect ----------
function markSelection() {
  for (const d of state.defects) d._marker?.setIcon(defectIcon(d));
  $$("#defect-list li, #worklist > li").forEach((li) => li.classList.toggle("selected", +li.dataset.id === state.selectedId));
  for (const [id, s] of state.shown) {
    const det = state.ride?.detections.find((x) => x.id === id);
    if (det) s.marker.setIcon(detectionIcon(det, { current: s.current }));
  }
}

function sparkline(history) {
  const perRide = Object.values(history.reduce((acc, h) => {
    if (!acc[h.ride_id] || h.severity > acc[h.ride_id].severity) acc[h.ride_id] = h;
    return acc;
  }, {})).sort((a, b) => a.recorded_at.localeCompare(b.recorded_at));
  if (perRide.length < 2) return "";
  const w = 340, h = 80, pad = 16;
  const xs = perRide.map((p) => new Date(p.recorded_at).getTime());
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const X = (x) => (x1 === x0 ? w / 2 : pad + ((x - x0) / (x1 - x0)) * (w - 2 * pad));
  const Y = (s) => h - pad - (s / 100) * (h - 2 * pad);
  const pts = perRide.map((p, i) => [X(xs[i]), Y(p.severity)]);
  const bands = [[75, COLORS.critical], [55, COLORS.high], [30, COLORS.medium]]
    .map(([v, c]) => `<line x1="${pad}" x2="${w - pad}" y1="${Y(v)}" y2="${Y(v)}" stroke="${c}" stroke-dasharray="3 4" opacity=".55"/>`).join("");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="Severity index per ride">${bands}
    <polyline fill="none" stroke="#1b1f24" stroke-width="2" points="${pts.map((p) => p.join(",")).join(" ")}"/>
    ${pts.map(([x, y], i) => `<circle cx="${x}" cy="${y}" r="4" fill="${COLORS[perRide[i].level]}" stroke="#fff" stroke-width="1.5"/>
      <text x="${x}" y="${y - 8}" font-size="10" text-anchor="middle">${sevNum(perRide[i].severity)}</text>`).join("")}
  </svg>`;
}

async function openDefect(id, { fly = true } = {}) {
  const d = await api(`/api/defects/${id}`);
  state.selectedId = id;
  markSelection();
  const synthetic = d.source === "synthetic";
  const reconstructed = d.history.some((h) => h.gps_source === "reconstructed");
  const meanConf = d.history.reduce((s, h) => s + h.conf, 0) / d.history.length;
  const total = d.breakdown.reduce((s, b) => s + b.points, 0);
  const box = $("#detail");
  box.hidden = false;
  box.innerHTML = `
    <div class="detail-head">
      <h2><span>${d.code} · ${esc(d.label)}</span><button class="btn ghost small" id="detail-close" aria-label="Close details">×</button></h2>
      <div class="tags">${levelBadge(d)} ${sourceTag(d.source)}</div>
    </div>
    ${synthetic ? '<section class="note" style="margin:0">This is a <b>synthetic demo defect</b>. Its images, confidence and severity were generated, not detected.</section>' : ""}
    <section>
      <h3>Priority <a href="#" data-open-guide="priority">How it's calculated</a></h3>
      <div class="big-score"><b>${d.priority.toFixed(1)}</b><span class="muted">/ 100 · rank ${d.rank} in this view</span></div>
      <p style="margin-top:8px"><b>${esc(d.action)}</b></p>
      <table class="breakdown">
        ${d.breakdown.map((b) => `<tr><td>${esc(b.label)}</td><td>${b.points >= 0 ? "+" : ""}${b.points.toFixed(1)}</td></tr>`).join("")}
        <tr class="total"><td>Total${total > 100 ? " (capped at 100)" : ""}</td><td>${d.priority.toFixed(1)}</td></tr>
      </table>
    </section>
    <section>
      <h3>Evidence ${synthetic ? "<span class=\"tag synthetic\">illustrations</span>" : ""}</h3>
      <div class="evidence">${d.history.map((h) => `<figure>
        ${h.crop_url ? `<img src="${h.crop_url}" alt="${esc(h.label)} on ${fmtDate(h.recorded_at)}">` : '<div class="empty">No image</div>'}
        <figcaption>${fmtShort(h.recorded_at)} · ${pct(h.conf)} conf.</figcaption></figure>`).join("")}</div>
    </section>
    <section>
      <h3>Assessment <a href="#" data-open-guide="severity">What these mean</a></h3>
      <dl class="kv">
        <dt>Severity index</dt><dd>${sevNum(d.severity)} / 100 · ${cap(d.level)} <span class="muted">(relative, uncalibrated)</span></dd>
        <dt>Mean confidence</dt><dd>${pct(meanConf)} over ${d.history.length} detection(s)</dd>
        <dt>Confirmations</dt><dd>${d.n_obs} independent ride${d.n_obs === 1 ? "" : "s"}</dd>
        <dt>Change</dt><dd>${esc(d.trend_label)}</dd>
        <dt>Status</dt><dd>${d.status === "active" ? "Active" : `Possibly repaired — missed on ${d.missed_passes} later pass(es)`}</dd>
      </dl>
    </section>
    <section>
      <h3>Location</h3>
      <dl class="kv">
        <dt>Coordinates</dt><dd class="mono">${d.lat.toFixed(6)}, ${d.lon.toFixed(6)}</dd>
        <dt>Positioning</dt><dd>${reconstructed ? "From reconstructed GPS — approximate (±10–20 m)" : synthetic ? "Simulated" : "Phone GPS + camera projection"}</dd>
      </dl>
      <p class="note"><a href="${d.maps_url}" target="_blank" rel="noopener">Open in Google Maps ↗</a></p>
    </section>
    <section>
      <h3>Observation history</h3>
      ${sparkline(d.history)}
      <table class="obs">
        <thead><tr><th>Date</th><th>Ride</th><th>Severity</th><th>Conf.</th><th></th></tr></thead>
        <tbody>${d.history.map((h) => `<tr>
          <td>${fmtShort(h.recorded_at)}</td><td>${esc(h.ride_name)}</td>
          <td>${sevNum(h.severity)} <span class="badge sev-${h.level}">${h.level}</span></td><td>${pct(h.conf)}</td>
          <td><button class="btn small" data-replay="${h.ride_id}" data-t="${h.t}" title="Open this moment in ride replay">▶ ${fmtTime(h.t)}</button></td>
        </tr>`).join("")}</tbody>
      </table>
    </section>
    <section>
      <h3>Why it ranks here</h3>
      <ul class="reasons">${d.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    </section>`;
  $("#detail-close").addEventListener("click", closeDetail);
  $$("[data-replay]", box).forEach((b) => b.addEventListener("click", () => replayAt(+b.dataset.replay, +b.dataset.t)));
  $$("[data-open-guide]", box).forEach(bindGuideLink);

  // Bring the defect into view. In the defect layer it may be inside a cluster.
  if (state.tab === "replay") {
    if (fly) map.setView([d.lat, d.lon], Math.max(map.getZoom(), 17));
  } else {
    const local = state.defects.find((x) => x.id === id);
    if (local && defectLayer.hasLayer(local._marker)) defectLayer.zoomToShowLayer(local._marker, () => fly && map.panTo([d.lat, d.lon]));
    else if (fly) map.setView([d.lat, d.lon], 17);
  }
  map.invalidateSize();
}
window.openDefect = openDefect;

function closeDetail() {
  state.selectedId = null;
  $("#detail").hidden = true;
  markSelection();
  map.invalidateSize();
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#detail").hidden && !document.querySelector("dialog[open]")) closeDetail(); });

async function replayAt(rideId, t) {
  const ride = state.rides.find((r) => r.id === rideId);
  if (ride && state.source !== "all" && ride.source !== state.source) await setSource("all");
  switchTab("replay");
  if (state.ride?.id !== rideId) await openRide(rideId);
  // Seek to the detection moment itself so its marker is on the map (it appears at t).
  seek(t);
  const [lat, lon] = positionAt(t);
  map.setView([lat, lon], 17);
}

// ---------- scoring guide ----------
const guide = $("#guide");
$("#guide-btn").addEventListener("click", () => guide.showModal());
$("#guide-close").addEventListener("click", () => guide.close());
guide.addEventListener("click", (e) => { if (e.target === guide) guide.close(); });
function bindGuideLink(a) {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    guide.showModal();
    const sec = $(`#g-${a.dataset.openGuide}`);
    sec?.scrollIntoView({ block: "start" });
    sec?.classList.remove("flash-target"); void sec?.offsetWidth; sec?.classList.add("flash-target");
  });
}
$$("[data-open-guide]").forEach(bindGuideLink);

// ---------- boot ----------
(async () => {
  await Promise.all([loadStats(), loadDefects()]);
  await loadRides();
})();
