// ESCA LoRA Lab — LoRA library panel inside ComfyUI.
// Catalog browser with purpose facets, drag-and-drop / one-click send into LoRA Manager
// loaders, a graph compatibility guard, and controls for the preview studio.
import { app } from "../../scripts/app.js";

const API = "/esca-lora";
const LM_LOADERS = new Set(["Lora Loader (LoraManager)", "Lora Stacker (LoraManager)"]);
const DRAG_TYPE = "application/x-esca-lora";
const MODEL_WIDGETS = { unet_name: "diffusion_models", ckpt_name: "checkpoints", model_name: "diffusion_models" };
const ARCH_LABEL = {
  "flux2-klein-9b": "Klein 9B", "flux2-klein-4b": "Klein 4B", "flux2-dev": "Flux.2 Dev", "flux1": "Flux.1",
  "sdxl": "SDXL", "sd1.5": "SD 1.5", "z-image": "Z-Image",
};

const STRINGS = {
  en: {
    title: "LoRA library", tabTooltip: "ESCA LoRA Lab — LoRA library", search: "Search: name, trigger, file…",
    compatible: "compatible with the model", rescan: "Rescan LoRA folders",
    syncTip: "Write categories and architecture into LoRA Manager", sync: "Sync", synced: "Everything is already in sync",
    syncAsk: (n) => `Update LoRA Manager metadata for ${n} LoRAs?`, syncDone: (a, e) => `Updated ${a}, errors ${e}`,
    of: (n, t) => `${n} of ${t}`, all: "All", copyTrigger: "Copy trigger", copied: "Copied", toGraph: "To graph",
    mb: "MB", noLm: "LoRA Manager not found", needLm: "A “Lora Loader (LoraManager)” node is required",
    added: "LoRA added", addedTo: (t, id) => `${t} → node #${id}`,
    mismatch: "LoRA does not fit the model", ignored: (t) => `${t} — the LoRA will be ignored`,
    mismatchText: (t, a, m) => `${t}: ${a} ≠ model ${m}`,
    noModel: "No model found in the graph — showing all LoRAs.",
    allOk: (m) => `Graph model: ${m}. All LoRAs in the graph are compatible.`,
    bad: (m, n) => `Model: ${m}. Incompatible LoRAs: ${n}. `,
    studio: "Preview studio", plan: "Plan", start: "Start", pause: "Pause", resume: "Resume", stop: "Stop",
    retry: "Retry failed", studioHint: "Click to refresh the status.",
    emptyQueue: "The queue is empty — press “Plan” first",
    startAsk: (n) => `Start rendering previews? ${n} images are queued. ` +
      "The studio yields the GPU to your own jobs and stops with “Stop”.",
    states: { idle: "stopped", running: "running", paused: "paused", stopping: "stopping" },
    statusLine: (s, d, q, f) => `State: ${s}. Done ${d}, queued ${q}, failed ${f}.`,
    now: (n, s, k) => `Now: ${n} · ${s} · ${k}`, baseline: "baseline scene",
    eta: (a, h) => `≈ ${a} s per image, ≈ ${h} h left`, lastError: (e) => `Last error: ${e}`,
    catalogFailed: (e) => `Catalog not loaded: ${e}`,
  },
  ru: {
    title: "Библиотека LoRA", tabTooltip: "ESCA LoRA Lab — библиотека LoRA", search: "Поиск: название, триггер, файл…",
    compatible: "совместимые с моделью", rescan: "Пересканировать папки LoRA",
    syncTip: "Записать категории и архитектуру в LoRA Manager", sync: "Синхронизация", synced: "Всё уже синхронизировано",
    syncAsk: (n) => `Обновить метаданные LoRA Manager у ${n} LoRA?`, syncDone: (a, e) => `Обновлено ${a}, ошибок ${e}`,
    of: (n, t) => `${n} из ${t}`, all: "Все", copyTrigger: "Копировать триггер", copied: "Скопировано", toGraph: "В граф",
    mb: "МБ", noLm: "LoRA Manager не найден", needLm: "Нужен узел «Lora Loader (LoraManager)»",
    added: "LoRA добавлена", addedTo: (t, id) => `${t} → узел #${id}`,
    mismatch: "LoRA не подходит к модели", ignored: (t) => `${t} — LoRA будет проигнорирована`,
    mismatchText: (t, a, m) => `${t}: ${a} ≠ модель ${m}`,
    noModel: "Модель в графе не найдена — показаны все LoRA.",
    allOk: (m) => `Модель графа: ${m}. Все LoRA в графе совместимы.`,
    bad: (m, n) => `Модель: ${m}. Несовместимых LoRA: ${n}. `,
    studio: "Студия превью", plan: "Запланировать", start: "Старт", pause: "Пауза", resume: "Продолжить", stop: "Стоп",
    retry: "Повторить ошибки", studioHint: "Нажмите, чтобы обновить статус.",
    emptyQueue: "Очередь пуста — сначала «Запланировать»",
    startAsk: (n) => `Запустить генерацию превью? В очереди ${n} изображений. ` +
      "Студия уступает GPU вашим задачам и останавливается кнопкой «Стоп».",
    states: { idle: "остановлена", running: "работает", paused: "на паузе", stopping: "останавливается" },
    statusLine: (s, d, q, f) => `Состояние: ${s}. Готово ${d}, в очереди ${q}, ошибок ${f}.`,
    now: (n, s, k) => `Сейчас: ${n} · ${s} · ${k}`, baseline: "базовая сцена",
    eta: (a, h) => `≈ ${a} с на изображение, осталось ≈ ${h} ч`, lastError: (e) => `Последняя ошибка: ${e}`,
    catalogFailed: (e) => `Каталог не загружен: ${e}`,
  },
};

function detectLang() {
  let loc = "";
  try { loc = app.extensionManager?.setting?.get?.("Comfy.Locale") || ""; } catch { /* older frontends */ }
  return String(loc || navigator.language || "en").toLowerCase().startsWith("ru") ? "ru" : "en";
}
// resolved in setup(), when the settings store is available
let LANG = "en";
let T = STRINGS.en;

const state = {
  entries: [], categories: [], byName: new Map(), byRel: new Map(),
  query: "", cat: null, compatOnly: true, nsfw: false,
  modelArch: null, warnings: [], seenWarnings: new Set(), signature: "", el: null,
};

const archOf = (a) => ARCH_LABEL[a] || a;
const norm = (s) => String(s || "").replace(/\\/g, "/").toLowerCase();

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
const postJSON = (url, body) =>
  getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function toast(severity, summary, detail) {
  app.extensionManager?.toast?.add({ severity, summary, detail, life: 6000 });
}

async function loadCatalog() {
  const data = await getJSON(`${API}/catalog?lang=${LANG}`);
  state.entries = data.entries;
  state.categories = data.categories;
  state.byName = new Map(data.entries.map((e) => [e.name.toLowerCase(), e]));
  state.byRel = new Map(data.entries.map((e) => [norm(e.rel), e]));
}

// ---------- graph helpers --------------------------------------------------------------
function allNodes() {
  return app.graph?._nodes || [];
}

function canvasCenter() {
  const c = app.canvas;
  const ds = c.ds;
  return [(-ds.offset[0] + c.canvas.width / 2 / ds.scale), (-ds.offset[1] + c.canvas.height / 2 / ds.scale)];
}

function createLmLoader(pos) {
  const node = LiteGraph.createNode("Lora Loader (LoraManager)");
  if (!node) return null;
  node.pos = pos;
  app.graph.add(node);
  return node;
}

async function sendToGraph(entry, strength = 1.0, targetNode = null) {
  const code = `<lora:${entry.name}:${strength}>`;
  let targets = targetNode ? [targetNode] : Object.values(app.canvas.selected_nodes || {})
    .filter((n) => LM_LOADERS.has(n.comfyClass));
  if (!targets.length) targets = allNodes().filter((n) => LM_LOADERS.has(n.comfyClass)).slice(0, 1);
  if (!targets.length) {
    const node = createLmLoader(canvasCenter());
    if (!node) {
      toast("warn", T.noLm, T.needLm);
      return;
    }
    targets = [node];
  }
  await postJSON("/api/lm/update-lora-code", { node_ids: targets.map((n) => n.id), lora_code: code, mode: "append" });
  toast("success", T.added, T.addedTo(entry.title, targets[0].id));
}

function loraRefsInGraph() {
  const refs = [];
  for (const node of allNodes()) {
    for (const w of node.widgets || []) {
      const v = w.value;
      if (typeof v === "string" && w.name === "lora_name" && v !== "None") {
        refs.push({ node, value: v });
      } else if (typeof v === "string" && (w.name === "text" || w.name === "lora_syntax") && v.includes("<lora:")) {
        for (const m of v.matchAll(/<lora:([^:>]+)(?::[^>]*)?>/g)) refs.push({ node, value: m[1] });
      } else if (v && typeof v === "object" && typeof v.lora === "string" && v.on !== false) {
        refs.push({ node, value: v.lora });                      // rgthree Power Lora Loader rows
      } else if (Array.isArray(v)) {
        for (const item of v) {                                  // LoRA Manager "loras" widget rows
          if (item && typeof item.name === "string" && item.active !== false) refs.push({ node, value: item.name });
        }
      }
    }
  }
  return refs;
}

function modelRefsInGraph() {
  const out = [];
  for (const node of allNodes()) {
    for (const w of node.widgets || []) {
      if (typeof w.value === "string" && MODEL_WIDGETS[w.name] && /\.(safetensors|gguf|ckpt|sft)$/i.test(w.value)) {
        out.push({ node, name: w.value, kind: w.value.toLowerCase().endsWith(".gguf") ? "unet_gguf" : MODEL_WIDGETS[w.name] });
      }
    }
  }
  return out;
}

function findEntry(value) {
  const v = norm(value);
  return state.byRel.get(v) || state.byRel.get(v.replace(/^[^/]+\//, "")) ||
    [...state.byRel.entries()].find(([rel]) => v.endsWith(rel))?.[1] ||
    state.byName.get(v.split("/").pop().replace(/\.safetensors$/, ""));
}

const archCache = new Map();
async function modelArch(ref) {
  const key = `${ref.kind}:${ref.name}`;
  if (!archCache.has(key)) {
    const r = await getJSON(`${API}/model-arch?name=${encodeURIComponent(ref.name)}&kind=${ref.kind}`).catch(() => ({ arch: "unknown" }));
    archCache.set(key, r.arch);
  }
  return archCache.get(key);
}

async function checkGraph(force = false) {
  if (!app.graph || !state.entries.length) return;
  const loras = loraRefsInGraph();
  const models = modelRefsInGraph();
  const sig = JSON.stringify([loras.map((r) => r.value), models.map((m) => m.name)]);
  if (!force && sig === state.signature) return;
  state.signature = sig;
  const archs = [...new Set((await Promise.all(models.map(modelArch))).filter((a) => a && a !== "unknown"))];
  state.modelArch = archs[0] || null;
  const warnings = [];
  if (state.modelArch) {
    for (const ref of loras) {
      const e = findEntry(ref.value);
      if (e && e.arch !== state.modelArch && !["unknown", "unreadable"].includes(e.arch)) {
        warnings.push({ node: ref.node, entry: e, text: T.mismatchText(e.title, archOf(e.arch), archOf(state.modelArch)) });
      }
    }
  }
  state.warnings = warnings;
  for (const w of warnings) {
    const key = `${w.node.id}:${w.entry.name}`;
    if (!state.seenWarnings.has(key)) {
      state.seenWarnings.add(key);
      toast("warn", T.mismatch, T.ignored(w.text));
    }
  }
  renderGuard();
  if (state.compatOnly) renderList();
}

// ---------- UI -------------------------------------------------------------------------
const CSS = `
.esca-lab{display:flex;flex-direction:column;height:100%;font-size:13px;color:var(--fg-color);}
.esca-lab header{padding:10px 10px 6px;border-bottom:1px solid var(--border-color);}
.esca-lab h3{margin:0 0 6px;font-size:14px;font-weight:600;display:flex;gap:6px;align-items:center;}
.esca-lab h3 small{color:var(--descrip-text);font-weight:400;margin-left:auto;}
.esca-lab input[type=search]{width:100%;box-sizing:border-box;padding:6px 8px;border-radius:6px;
  border:1px solid var(--border-color);background:var(--comfy-input-bg);color:var(--input-text,var(--fg-color));}
.esca-row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:6px;color:var(--descrip-text);}
.esca-row label{display:inline-flex;gap:4px;align-items:center;cursor:pointer;}
.esca-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;}
.esca-chips button,.esca-btn{border:1px solid var(--border-color);background:var(--comfy-input-bg);color:var(--fg-color);
  border-radius:999px;padding:2px 9px;cursor:pointer;font-size:12px;}
.esca-chips button[aria-pressed=true]{background:var(--p-primary-color,#3e9a86);border-color:transparent;color:#fff;}
.esca-guard{margin:6px 10px 0;padding:6px 8px;border-radius:6px;font-size:12px;}
.esca-guard.ok{background:color-mix(in srgb,#3e9a86 14%,transparent);}
.esca-guard.warn{background:color-mix(in srgb,#d9952b 18%,transparent);}
.esca-guard a{cursor:pointer;text-decoration:underline;}
.esca-list{flex:1;overflow:auto;padding:6px 10px 10px;display:flex;flex-direction:column;gap:6px;}
.esca-card{border:1px solid var(--border-color);border-radius:8px;padding:7px 8px;background:var(--comfy-menu-bg);cursor:grab;}
.esca-card .t{font-weight:600;overflow-wrap:anywhere;}
.esca-card .m{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px;}
.esca-card .c{font-size:11px;padding:0 6px;border-radius:5px;background:var(--comfy-input-bg);color:var(--descrip-text);}
.esca-card .c.bad{background:color-mix(in srgb,#d9432f 22%,transparent);color:var(--fg-color);}
.esca-card .n{margin-top:4px;font-size:12px;color:var(--descrip-text);}
.esca-card .trig{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;}
.esca-card .trig button{font:11px/1.3 monospace;border:1px dashed var(--border-color);background:none;color:var(--fg-color);
  border-radius:5px;padding:1px 5px;cursor:copy;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.esca-card .a{display:flex;gap:6px;align-items:center;margin-top:5px;}
.esca-card .a a{color:var(--descrip-text);font-size:11px;}
.esca-card .a .add{margin-left:auto;}
.esca-studio{border-top:1px solid var(--border-color);padding:8px 10px;font-size:12px;}
.esca-studio summary{cursor:pointer;font-weight:600;}
.esca-studio .bar{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;}
.esca-studio .st{color:var(--descrip-text);margin-top:4px;line-height:1.4;}
`;

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

function filtered() {
  const q = state.query.trim().toLowerCase();
  return state.entries.filter((e) =>
    (state.nsfw || !e.nsfw) &&
    (!state.cat || e.category === state.cat) &&
    (!state.compatOnly || !state.modelArch || e.arch === state.modelArch) &&
    (!q || [e.title, e.name, e.rel, e.category_label, (e.triggers || []).join(" "), e.creator].join(" ").toLowerCase().includes(q)));
}

function renderList() {
  const box = state.el?.querySelector(".esca-list");
  if (!box) return;
  const list = filtered().sort((a, b) => a.title.localeCompare(b.title, LANG));
  box.replaceChildren(...list.slice(0, 400).map(card));
  state.el.querySelector(".esca-count").textContent = T.of(list.length, state.entries.length);
  renderChips();
}

function renderChips() {
  const base = state.entries.filter((e) => (state.nsfw || !e.nsfw) &&
    (!state.compatOnly || !state.modelArch || e.arch === state.modelArch));
  const counts = {};
  for (const e of base) counts[e.category] = (counts[e.category] || 0) + 1;
  const box = state.el.querySelector(".esca-chips");
  const chip = (key, label, n) => h("button", {
    "aria-pressed": String(state.cat === key),
    onclick: () => { state.cat = state.cat === key ? null : key; renderList(); },
  }, n == null ? label : `${label} · ${n}`);
  box.replaceChildren(chip(null, T.all, base.length),
    ...state.categories.filter((c) => counts[c.key]).map((c) => chip(c.key, c.label, counts[c.key])));
}

function card(e) {
  const bad = state.modelArch && e.arch !== state.modelArch;
  const strength = 1.0;
  return h("div", { class: "esca-card", draggable: "true", title: e.rel,
    ondragstart: (ev) => ev.dataTransfer.setData(DRAG_TYPE, JSON.stringify({ name: e.name, strength })) },
    h("div", { class: "t" }, e.title + (e.version ? ` · ${e.version}` : "")),
    h("div", { class: "m" },
      h("span", { class: "c" }, e.category_label),
      h("span", { class: `c${bad ? " bad" : ""}` }, archOf(e.arch)),
      e.trained_on ? h("span", { class: "c" }, e.trained_on) : null,
      e.fmt !== "lora" ? h("span", { class: "c" }, e.fmt.toUpperCase()) : null,
      e.nsfw ? h("span", { class: "c bad" }, "NSFW") : null),
    (e.triggers || []).length ? h("div", { class: "trig" }, ...e.triggers.slice(0, 4).map((t) =>
      h("button", { title: T.copyTrigger, onclick: () => navigator.clipboard.writeText(t)
        .then(() => toast("info", T.copied, t)) }, t))) : null,
    e.note ? h("div", { class: "n" }, e.note) : null,
    h("div", { class: "a" },
      h("a", { href: `/loras?search=${encodeURIComponent(e.name)}`, target: "_blank" }, "LoRA Manager"),
      e.civitai_url ? h("a", { href: e.civitai_url, target: "_blank", rel: "noopener" }, "Civitai") : null,
      h("span", { class: "c" }, `r${e.rank ?? "—"} · ${e.size_mb} ${T.mb}`),
      h("button", { class: "esca-btn add", onclick: () => sendToGraph(e, strength) }, T.toGraph)));
}

function renderGuard() {
  const box = state.el?.querySelector(".esca-guard");
  if (!box) return;
  if (!state.modelArch) {
    box.className = "esca-guard";
    box.textContent = T.noModel;
    return;
  }
  if (!state.warnings.length) {
    box.className = "esca-guard ok";
    box.textContent = T.allOk(archOf(state.modelArch));
    return;
  }
  box.className = "esca-guard warn";
  box.replaceChildren(T.bad(archOf(state.modelArch), state.warnings.length),
    ...state.warnings.slice(0, 4).map((w) => h("div", {},
      h("a", { onclick: () => { app.canvas.centerOnNode?.(w.node); app.canvas.selectNode?.(w.node); } }, `#${w.node.id}`),
      ` ${w.text}`)));
}

async function studioCall(action, body) {
  try {
    const s = action === "status" ? await getJSON(`${API}/previews/status`) : await postJSON(`${API}/previews/${action}`, body);
    renderStudio(s);
  } catch (err) {
    toast("error", T.studio, String(err));
  }
}

function renderStudio(s) {
  const st = state.el?.querySelector(".esca-studio .st");
  if (!st || !s) return;
  const c = s.counts || {};
  st.replaceChildren(
    h("div", {}, T.statusLine(T.states[s.state] || s.state, c.done || 0, (c.pending || 0) + (c.running || 0), c.failed || 0)),
    s.current ? h("div", {}, T.now(s.current.lora_name || T.baseline, s.current.scene, s.current.strength)) : "",
    s.eta_minutes != null ? h("div", {}, T.eta(s.avg_seconds, Math.round(s.eta_minutes / 6) / 10)) : "",
    s.last_error ? h("div", {}, T.lastError(s.last_error)) : "");
}

async function syncWithLm() {
  const plan = await getJSON(`${API}/sync`);
  if (!plan.to_change) return toast("info", T.sync, T.synced);
  if (!confirm(T.syncAsk(plan.to_change))) return;
  const r = await postJSON(`${API}/sync`, { confirm: true });
  toast(r.errors?.length ? "warn" : "success", T.sync, T.syncDone(r.applied, r.errors?.length || 0));
}

async function startStudio() {
  const s = await getJSON(`${API}/previews/status`);
  const left = (s.counts?.pending || 0) + (s.counts?.running || 0);
  if (!left) return toast("info", T.studio, T.emptyQueue);
  if (confirm(T.startAsk(left))) studioCall("start", { polite: true });
}

function mount(el) {
  state.el = el;
  el.replaceChildren(
    h("style", {}, CSS),
    h("div", { class: "esca-lab" },
      h("header", {},
        h("h3", {}, T.title, h("small", { class: "esca-count" }, "")),
        h("input", { type: "search", placeholder: T.search,
          oninput: (ev) => { state.query = ev.target.value; renderList(); } }),
        h("div", { class: "esca-row" },
          h("label", {}, h("input", { type: "checkbox", checked: state.compatOnly,
            onchange: (ev) => { state.compatOnly = ev.target.checked; renderList(); } }), T.compatible),
          h("label", {}, h("input", { type: "checkbox", checked: state.nsfw,
            onchange: (ev) => { state.nsfw = ev.target.checked; renderList(); } }), "NSFW"),
          h("button", { class: "esca-btn", title: T.rescan,
            onclick: async () => { await postJSON(`${API}/catalog/rebuild`); await loadCatalog(); renderList(); } }, "⟳"),
          h("button", { class: "esca-btn", title: T.syncTip, onclick: syncWithLm }, "⇄ LM")),
        h("div", { class: "esca-chips" })),
      h("div", { class: "esca-guard" }, ""),
      h("div", { class: "esca-list" }),
      h("details", { class: "esca-studio" },
        h("summary", { onclick: () => studioCall("status") }, T.studio),
        h("div", { class: "bar" },
          h("button", { class: "esca-btn", onclick: () => studioCall("plan", {}) }, T.plan),
          h("button", { class: "esca-btn", onclick: startStudio }, T.start),
          h("button", { class: "esca-btn", onclick: () => studioCall("pause") }, T.pause),
          h("button", { class: "esca-btn", onclick: () => studioCall("resume") }, T.resume),
          h("button", { class: "esca-btn", onclick: () => studioCall("stop") }, T.stop),
          h("button", { class: "esca-btn", onclick: () => studioCall("retry-failed") }, T.retry)),
        h("div", { class: "st" }, T.studioHint))));
  loadCatalog().then(() => { renderList(); checkGraph(true); }).catch((err) =>
    toast("error", "ESCA LoRA Lab", T.catalogFailed(err)));
}

function installDrop() {
  const canvasEl = app.canvasEl || app.canvas?.canvas;
  if (!canvasEl || canvasEl.__escaDrop) return;
  canvasEl.__escaDrop = true;
  canvasEl.addEventListener("dragover", (ev) => {
    if (ev.dataTransfer?.types?.includes(DRAG_TYPE)) ev.preventDefault();
  });
  canvasEl.addEventListener("drop", async (ev) => {
    const raw = ev.dataTransfer?.getData(DRAG_TYPE);
    if (!raw) return;
    ev.preventDefault();
    ev.stopPropagation();
    const { name, strength } = JSON.parse(raw);
    const entry = state.byName.get(name.toLowerCase());
    if (!entry) return;
    const pos = app.canvas.convertEventToCanvasOffset(ev);
    let node = app.graph.getNodeOnPos(pos[0], pos[1]);
    if (!node || !LM_LOADERS.has(node.comfyClass)) node = createLmLoader(pos);
    if (node) await sendToGraph(entry, strength, node);
  }, true);
}

app.registerExtension({
  name: "esca.loraLab",
  async setup() {
    LANG = detectLang();
    T = STRINGS[LANG];
    app.extensionManager.registerSidebarTab({
      id: "esca-lora-lab",
      icon: "pi pi-images",
      title: "LoRA",
      tooltip: T.tabTooltip,
      type: "custom",
      render: (el) => mount(el),
    });
    installDrop();
    if (!state.entries.length) loadCatalog().catch(() => {});
    setInterval(() => { if (document.visibilityState === "visible") checkGraph().catch(() => {}); }, 2500);
  },
});
