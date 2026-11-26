(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const inputs = [...document.querySelectorAll('input[type="range"]')];
  const modelBtns = [...document.querySelectorAll(".model-btn")];
  const results = $("results");
  const ICONS = { green: "✅", lime: "👍", amber: "⚠️", red: "🛑" };

  let model = window.__INITIAL__.model_used;
  let inflight = false, dirty = false;
  const cache = new Map();                       // instant repeat values, no round-trip
  let last = window.__INITIAL__;

  // ---------------------------------------------------------------- inputs
  const fmt = (el) => (+el.value).toFixed(+el.dataset.decimals) + el.dataset.unit;
  function paintSlider(el) {
    const p = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty("--p", p + "%");
    $("out-" + el.dataset.key).textContent = fmt(el);
  }
  const payload = () => {
    const body = { model };
    inputs.forEach((el) => (body[el.dataset.key] = +(+el.value).toFixed(+el.dataset.decimals)));
    return body;
  };

  // ------------------------------------------------------------ sync engine
  // At most one request in flight; when it returns, send the newest slider state.
  // The UI therefore always converges on the latest values and never queues up.
  function sync() {
    dirty = true;
    if (!inflight) run();
  }
  async function run() {
    inflight = true; dirty = false;
    const body = payload(), key = JSON.stringify(body);
    try {
      let data = cache.get(key);
      if (!data) {
        const res = await fetch("/api/predict/", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: key,
        });
        if (!res.ok) throw new Error((await res.json()).error || res.statusText);
        data = await res.json();
        if (cache.size > 2000) cache.clear();
        cache.set(key, data);
      }
      if (!dirty) render(data);                  // skip stale frames
    } catch (err) {
      console.error("Prediction failed:", err);
    }
    inflight = false;
    if (dirty) run();
  }

  // ---------------------------------------------------------------- render
  function render(d) {
    last = d;
    if (d.model_used !== model) { model = d.model_used; paintModels(); }
    results.dataset.tone = d.tone;
    $("soh").textContent = d.soh.toFixed(2) + "%";
    $("rul").textContent = "~" + d.remaining_cycles;
    $("fill").style.height = d.soh + "%";
    $("badge-icon").textContent = ICONS[d.tone] || "✅";
    $("status").textContent = d.status;
    $("message").textContent = d.message;
    $("value").textContent = "$" + d.pack_value.toLocaleString("en-US");
    drawChart(d.curve);
  }
  function paintModels() {
    modelBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.model === model)));
  }

  // ----------------------------------------------------------------- chart
  const chartEl = $("chart");
  const NS = "http://www.w3.org/2000/svg";
  const X = [0, 100, 200, 300, 400, 500, 600];
  const YMIN = 40, YMAX = 102, M = { l: 52, r: 16, t: 12, b: 44 };

  function el(name, attrs, parent, text) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    parent && parent.appendChild(n);
    return n;
  }
  function drawChart(curve) {
    const W = chartEl.clientWidth || 700, H = chartEl.clientHeight || 300;
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const sx = (v) => M.l + (v / 600) * iw;
    const sy = (v) => M.t + (1 - (Math.max(YMIN, v) - YMIN) / (YMAX - YMIN)) * ih;

    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    for (const t of [40, 50, 60, 70, 80, 90, 100]) {
      el("line", { x1: M.l, x2: W - M.r, y1: sy(t), y2: sy(t), stroke: "#e6e9f2" }, svg);
      el("text", { x: M.l - 8, y: sy(t) + 4, "text-anchor": "end" }, svg, t + "%");
    }
    for (const x of X) {
      el("line", { x1: sx(x), x2: sx(x), y1: M.t, y2: M.t + ih, stroke: "#eef0f6" }, svg);
      el("text", { x: sx(x), y: M.t + ih + 18, "text-anchor": "middle" }, svg, x);
    }
    el("text", { x: M.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "title-axis" }, svg, "Charge Cycles");
    el("text", { transform: `translate(14 ${M.t + ih / 2}) rotate(-90)`, "text-anchor": "middle", class: "title-axis" }, svg, "State of Health (%)");

    const pts = curve.map((v, i) => [sx(X[i]), sy(v)]);
    const line = pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" ");
    el("path", { d: `${line} L${pts[pts.length - 1][0]} ${M.t + ih} L${pts[0][0]} ${M.t + ih} Z`, fill: "rgba(34,160,79,.10)" }, svg);
    el("line", { x1: M.l, x2: W - M.r, y1: sy(80), y2: sy(80), stroke: "#e0463e", "stroke-width": 1.6, "stroke-dasharray": "5 4" }, svg);
    el("path", { d: line, fill: "none", stroke: "#22a04f", "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);
    pts.forEach((p, i) => {
      const c = el("circle", { cx: p[0], cy: p[1], r: 5, fill: "#22a04f", stroke: "#fff", "stroke-width": 1.5 }, svg);
      el("title", {}, c, `Cycle ${X[i]}: ${curve[i].toFixed(2)}%`);
    });
    chartEl.replaceChildren(svg);
  }

  // ---------------------------------------------------------------- wiring
  inputs.forEach((el) => {
    paintSlider(el);
    el.addEventListener("input", () => { paintSlider(el); sync(); });
  });
  modelBtns.forEach((b) => b.addEventListener("click", () => {
    if (b.disabled || b.dataset.model === model) return;
    model = b.dataset.model; paintModels(); sync();
  }));
  $("predict").addEventListener("click", sync);
  new ResizeObserver(() => drawChart(last.curve)).observe(chartEl);
  drawChart(last.curve);
})();
