/**
 * Lightweight Newsprint-themed Three.js Swarm Theater for Edition No. 2.
 * Supports real-time Server-Sent Events (/api/stream) locally and static JSON replay on GitHub Pages.
 */

const CLUSTER_SLUGS = [
  "glm-5-3-flash",
  "gemma-4-31b-it",
  "qwen3-8-27b",
  "glm-5-2",
  "nemotron-sea-lion-v4-8-120b-a12b",
  "deepseek-v4-flash",
  "qwen-sea-lion-v5-397b-a17b-plus",
  "gemma-4-26b-a4b-it",
  "qwen3-5-397b",
];

const SHORT_NAMES = {
  "glm-5-3-flash": "glm-5.3-flash",
  "gemma-4-31b-it": "gemma-4-31b",
  "qwen3-8-27b": "qwen3.8-27b",
  "glm-5-2": "glm-5.2",
  "nemotron-sea-lion-v4-8-120b-a12b": "nemo-SL-120b",
  "deepseek-v4-flash": "deepseek-v4",
  "qwen-sea-lion-v5-397b-a17b-plus": "qwen-SL-397b",
  "gemma-4-26b-a4b-it": "gemma-4-26b",
  "qwen3-5-397b": "qwen3.5-397b",
};

const KIND_HEX = {
  coordinator_brief: 0x9a6314,
  cluster_spawn: 0x9a6314,
  resource_lookup: 0x1e5c3a,
  agent_exchange: 0x1b4d6e,
  consensus_vote: 0xc2781a,
  consensus_reached: 0x1e5c3a,
  shadow_patch: 0x1b4d6e,
  native_command: 0x8c1d24,
  native_generation: 0x8c1d24,
  coordinator_audit: 0x181613,
  run_complete: 0x1e5c3a,
};

function getThemePalette() {
  const isEv = document.documentElement.getAttribute("data-theme") === "evening";
  return isEv
    ? { bg: 0x161412, grid1: 0x3e3830, grid2: 0x2a2520, ring: 0x6f675b, edge: 0x8f8577, ink: 0xede6d8 }
    : { bg: 0xf5f1e6, grid1: 0xcbc2af, grid2: 0xe3dccb, ring: 0x8a8275, edge: 0x554f47, ink: 0x181613 };
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function topologyOffsets(n, topology) {
  const pts = [];
  if (topology === "star") {
    pts.push([0, 0.1, 0]);
    for (let i = 1; i < n; i++) {
      const a = ((i - 1) / Math.max(1, n - 1)) * Math.PI * 2;
      pts.push([Math.cos(a) * 1.45, 0, Math.sin(a) * 1.45]);
    }
  } else if (topology === "pipeline") {
    for (let i = 0; i < n; i++) {
      const x = -1.4 + (i * 2.8) / Math.max(1, n - 1);
      pts.push([x, 0, (i % 2 === 0 ? 0.35 : -0.35)]);
    }
  } else if (topology === "hierarchical") {
    pts.push([0, 0.55, -1.1]);
    for (let i = 1; i < n; i++) {
      const x = -1.3 + ((i - 1) * 2.6) / Math.max(1, n - 1);
      pts.push([x, 0, i === 1 && n >= 4 ? -0.15 : 0.95]);
    }
  } else {
    // ring / mesh
    for (let i = 0; i < n; i++) {
      const a = (i / Math.max(1, n)) * Math.PI * 2 - Math.PI / 2;
      pts.push([Math.cos(a) * 1.35, 0, Math.sin(a) * 1.35]);
    }
  }
  return pts;
}

class SwarmTheater {
  constructor(rootEl, opts = {}) {
    this.root = rootEl;
    this.dataPrefix = opts.dataPrefix || "../../data";
    this.liveOrigin = (opts.liveOrigin || "").replace(/\/$/, "");
    this.apiBase = null;
    this.eventsByLang = {};
    this.runsMeta = [];
    this.healthSnap = null;
    this.selectedLang = opts.defaultLang || "taq";
    this.selectedCluster = "ALL";
    this.cursor = 0;
    this.playing = true;
    this.followLive = false;
    this.liveAvailable = false;
    this.speed = 2;
    this.sse = null;
    this.pulses = [];
    this.clusterStates = {};
    this.labelNodes = [];
    this.camTarget = { x: 0, y: 0, z: 0 };
    this.camDesiredPos = { x: 0, y: 18, z: 22 };
    this.useThree = false;
    this.THREE = null;
    this._mountDom(opts.fullscreen);
    this._init();
  }

  _mountDom(fullscreen) {
    this.root.innerHTML = `
      <div class="lang-ribbon" id="stLangRibbon">
        <span style="font-weight:700;margin-right:4px">MULTI13:</span>
        <span style="color:var(--ink-soft)">Connecting to live telemetry...</span>
      </div>
      <div class="theater-toolbar">
        <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
          <label>Lang <select id="stLang"></select></label>
          <label>Cluster <select id="stCluster"><option value="ALL">All 9 Clusters</option></select></label>
          <button type="button" id="stPlay">Pause</button>
          <button type="button" id="stReplay">Restart</button>
          <button type="button" id="stLive" style="display:none">Follow Live</button>
          <button type="button" id="stConnectLive" style="display:none">Connect Live</button>
          <label>Speed
            <select id="stSpeed">
              <option value="1">1x</option>
              <option value="2" selected>2x</option>
              <option value="4">4x</option>
              <option value="8">8x</option>
            </select>
          </label>
          <button type="button" id="stAdminBtn" style="display:none">Admin</button>
        </div>
        <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:210px;max-width:360px">
          <input id="stScrub" type="range" min="0" max="0" value="0" style="flex:1" />
          <span id="stStepLbl" style="font-family:var(--mono);font-size:0.72rem;min-width:62px;text-align:right">0 / 0</span>
        </div>
      </div>
      <div class="admin-drawer" id="stAdminDrawer">
        <strong>Admin Trigger:</strong>
        <input id="stAdminTok" type="password" placeholder="SWARM_ADMIN_TOKEN" style="width:170px" />
        <button type="button" id="stTrigLang">Re-run Selected Lang</button>
        <button type="button" id="stTrigAll">Run Pending All-13</button>
        <button type="button" id="stTrigStop">Stop After Current</button>
        <span id="stAdminMsg" style="font-family:var(--mono);font-size:0.72rem"></span>
      </div>
      <div class="theater-wrap ${fullscreen ? "fullscreen" : ""}">
        <div>
          <div class="theater-stage-box" id="stStageBox">
            <canvas id="stCanvas"></canvas>
            <div class="theater-labels" id="stLabels"></div>
          </div>
          <div class="stage-legend">
            <span><span class="leg-dot" style="background:#9a6314"></span>Coordinator / Spawn</span>
            <span><span class="leg-dot" style="background:#1e5c3a"></span>Dict/Gram Lookup &amp; Consensus</span>
            <span><span class="leg-dot" style="background:#1b4d6e"></span>Agent Exchange &amp; Shadow DabaX</span>
            <span><span class="leg-dot" style="background:#8c1d24"></span>prompt_native.csv Directive</span>
            <span>Drag to orbit &bull; Scroll to zoom &bull; Click any cluster pill to focus</span>
          </div>
        </div>
        <div class="hud-col">
          <div class="hud-kpis">
            <div class="hud-kpi"><div class="k">Focus</div><div class="v" id="hkFocus">ALL</div></div>
            <div class="hud-kpi"><div class="k">Size / Topo</div><div class="v" id="hkTopo">-</div></div>
            <div class="hud-kpi"><div class="k">Round / Status</div><div class="v" id="hkRound">R0</div></div>
            <div class="hud-kpi"><div class="k">Exchanges</div><div class="v" id="hkEx">0</div></div>
            <div class="hud-kpi"><div class="k">Shadow Δ</div><div class="v" id="hkShadow">-</div></div>
            <div class="hud-kpi"><div class="k">Native DabaX</div><div class="v" id="hkNative">-</div></div>
          </div>
          <div class="cluster-strip" id="stClusterStrip"></div>
          <div class="exchange-feed" id="stFeed"></div>
        </div>
      </div>
    `;
    this.$lang = this.root.querySelector("#stLang");
    this.$cluster = this.root.querySelector("#stCluster");
    this.$play = this.root.querySelector("#stPlay");
    this.$replay = this.root.querySelector("#stReplay");
    this.$live = this.root.querySelector("#stLive");
    this.$connectLive = this.root.querySelector("#stConnectLive");
    this._statusTimer = null;
    if (this.$connectLive) {
      this.$connectLive.addEventListener("click", () => this._onConnectLiveClick());
    }
    this.$speed = this.root.querySelector("#stSpeed");
    this.$scrub = this.root.querySelector("#stScrub");
    this.$stepLbl = this.root.querySelector("#stStepLbl");
    this.$stageBox = this.root.querySelector("#stStageBox");
    this.$canvas = this.root.querySelector("#stCanvas");
    this.$labels = this.root.querySelector("#stLabels");
    this.$strip = this.root.querySelector("#stClusterStrip");
    this.$feed = this.root.querySelector("#stFeed");
    this.$langRibbon = this.root.querySelector("#stLangRibbon");
    this.$adminBtn = this.root.querySelector("#stAdminBtn");
    this.$adminDrawer = this.root.querySelector("#stAdminDrawer");
    this.$adminTok = this.root.querySelector("#stAdminTok");
    this.$adminMsg = this.root.querySelector("#stAdminMsg");

    const savedTok = sessionStorage.getItem("swarm_admin_tok") || "";
    if (savedTok) this.$adminTok.value = savedTok;
    this.$adminBtn.addEventListener("click", () => {
      this.$adminDrawer.classList.toggle("open");
    });
    this.$adminTok.addEventListener("input", () => {
      sessionStorage.setItem("swarm_admin_tok", this.$adminTok.value);
    });
    this.root.querySelector("#stTrigLang").addEventListener("click", () =>
      this._adminPost("/api/admin/trigger", { langs: [this.selectedLang], resume: false })
    );
    this.root.querySelector("#stTrigAll").addEventListener("click", () =>
      this._adminPost("/api/admin/trigger", { langs: "all", resume: true })
    );
    this.root.querySelector("#stTrigStop").addEventListener("click", () =>
      this._adminPost("/api/admin/stop", {})
    );

    for (const slug of CLUSTER_SLUGS) {
      const opt = document.createElement("option");
      opt.value = slug;
      opt.textContent = SHORT_NAMES[slug] || slug;
      this.$cluster.appendChild(opt);
    }

    this.$lang.addEventListener("change", () => this._switchLang(this.$lang.value));
    this.$cluster.addEventListener("change", () => this.focusCluster(this.$cluster.value));
    this.$play.addEventListener("click", () => {
      this.playing = !this.playing;
      this.$play.textContent = this.playing ? "Pause" : "Play";
    });
    this.$replay.addEventListener("click", () => {
      this.followLive = false;
      this.seekTo(0);
      this.playing = true;
      this.$play.textContent = "Pause";
    });
    this.$live.addEventListener("click", () => {
      this.followLive = !this.followLive;
      this.$live.classList.toggle("active", this.followLive);
      if (this.followLive) this._connectSse(this.selectedLang);
      else if (this.sse) { this.sse.close(); this.sse = null; }
    });
    this.$speed.addEventListener("change", () => {
      this.speed = Number(this.$speed.value) || 2;
    });
    this.$scrub.addEventListener("input", () => {
      this.playing = false;
      this.$play.textContent = "Play";
      this.seekTo(Number(this.$scrub.value));
    });
  }

  async _init() {
    await this._initRenderer();
    await this._loadDataSource();
    this._bindThemeObserver();
    this._lastStepTs = performance.now();
    this._animate();
  }

  async _initRenderer() {
    try {
      const THREE = await import("https://esm.sh/three@0.170.0");
      const { OrbitControls } = await import("https://esm.sh/three@0.170.0/examples/jsm/controls/OrbitControls.js");
      this.THREE = THREE;
      const renderer = new THREE.WebGLRenderer({ canvas: this.$canvas, antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(this.$stageBox.clientWidth, this.$stageBox.clientHeight, false);
      const pal = getThemePalette();
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(pal.bg);
      const camera = new THREE.PerspectiveCamera(46, this.$stageBox.clientWidth / this.$stageBox.clientHeight, 0.1, 140);
      camera.position.set(0, 18, 22);
      const controls = new OrbitControls(camera, this.$canvas);
      controls.enableDamping = true;
      controls.maxPolarAngle = Math.PI / 2.08;

      const hemi = new THREE.HemisphereLight(0xffffff, pal.grid1, 1.05);
      const dir = new THREE.DirectionalLight(0xffffff, 0.75);
      dir.position.set(12, 22, 10);
      scene.add(hemi, dir);

      const grid = new THREE.GridHelper(34, 24, pal.grid1, pal.grid2);
      grid.position.y = -0.35;
      scene.add(grid);

      this.renderer = renderer;
      this.scene = scene;
      this.camera = camera;
      this.controls = controls;
      this.grid = grid;
      this._buildStaticStage();
      this.useThree = true;

      window.addEventListener("resize", () => {
        const w = this.$stageBox.clientWidth;
        const h = this.$stageBox.clientHeight;
        if (!w || !h) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
      });
    } catch (err) {
      console.warn("WebGL/Three fallback to 2D Canvas:", err);
      this.useThree = false;
    }
  }

  _buildStaticStage() {
    const THREE = this.THREE;
    const pal = getThemePalette();
    this.anchorPositions = {
      coordinator: new THREE.Vector3(0, 0.5, 0),
      "res:baseline": new THREE.Vector3(-12.5, 0.4, -7.5),
      "res:shadow_dabax": new THREE.Vector3(12.5, 0.4, -7.5),
      "res:prompt_native": new THREE.Vector3(0, 0.4, -12.8),
    };
    this.agentPositions = {};
    this.clusterGroups = {};

    // Coordinator orb + ring
    const cMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 18, 12),
      new THREE.MeshLambertMaterial({ color: 0x9a6314 })
    );
    cMesh.position.copy(this.anchorPositions.coordinator);
    this.scene.add(cMesh);
    const cHalo = new THREE.Mesh(
      new THREE.RingGeometry(1.4, 1.52, 48),
      new THREE.MeshBasicMaterial({ color: pal.ink, side: THREE.DoubleSide })
    );
    cHalo.rotation.x = -Math.PI / 2;
    this.scene.add(cHalo);
    this._registerLabel("coordinator", this.anchorPositions.coordinator, "Coordinator · Nemotron-550B", "coord");

    // Resource monoliths
    const rBase = new THREE.Mesh(
      new THREE.BoxGeometry(1.15, 1.15, 1.15),
      new THREE.MeshLambertMaterial({ color: 0x1e5c3a })
    );
    rBase.position.copy(this.anchorPositions["res:baseline"]);
    const rShadow = new THREE.Mesh(
      new THREE.CylinderGeometry(0.65, 0.65, 1.25, 16),
      new THREE.MeshLambertMaterial({ color: 0x1b4d6e })
    );
    rShadow.position.copy(this.anchorPositions["res:shadow_dabax"]);
    const rNative = new THREE.Mesh(
      new THREE.ConeGeometry(0.75, 1.35, 4),
      new THREE.MeshLambertMaterial({ color: 0x8c1d24 })
    );
    rNative.position.copy(this.anchorPositions["res:prompt_native"]);
    this.scene.add(rBase, rShadow, rNative);
    this._registerLabel("res:baseline", this.anchorPositions["res:baseline"], "baseline.gram + .dict");
    this._registerLabel("res:shadow_dabax", this.anchorPositions["res:shadow_dabax"], "Shadow DabaX");
    this._registerLabel("res:prompt_native", this.anchorPositions["res:prompt_native"], "prompt_native.csv");

    // 9 Cluster Rings
    CLUSTER_SLUGS.forEach((slug, idx) => {
      const angle = (idx / CLUSTER_SLUGS.length) * Math.PI * 2 + Math.PI / 4;
      const rx = Math.cos(angle) * 8.2;
      const rz = Math.sin(angle) * 7.6 + 0.8;
      this.anchorPositions[slug] = new THREE.Vector3(rx, 0.15, rz);

      const grp = new THREE.Group();
      grp.position.set(rx, 0, rz);
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(1.85, 1.94, 36),
        new THREE.MeshBasicMaterial({ color: pal.ring, side: THREE.DoubleSide })
      );
      ring.rotation.x = -Math.PI / 2;
      grp.add(ring);
      this.scene.add(grp);
      this.clusterGroups[slug] = { group: grp, ring, dynGroup: new THREE.Group(), center: new THREE.Vector3(rx, 0, rz) };
      grp.add(this.clusterGroups[slug].dynGroup);
      this._registerLabel(slug, new THREE.Vector3(rx, 0.35, rz - 1.95), SHORT_NAMES[slug], "cluster", slug);
    });

    // Event chain strip on the left rim
    this.chainGroup = new THREE.Group();
    this.scene.add(this.chainGroup);
  }

  _registerLabel(key, vec3, text, role = "", clusterSlug = "") {
    const el = document.createElement("div");
    el.className = `node-pill ${role}`;
    el.innerHTML = esc(text);
    if (clusterSlug) {
      el.addEventListener("click", () => this.focusCluster(clusterSlug));
    }
    this.$labels.appendChild(el);
    this.labelNodes.push({ key, vec3, el, baseText: text, clusterSlug });
  }

  _bindThemeObserver() {
    const obs = new MutationObserver(() => {
      if (!this.useThree) return;
      const pal = getThemePalette();
      this.scene.background = new this.THREE.Color(pal.bg);
      this.scene.remove(this.grid);
      this.grid = new this.THREE.GridHelper(34, 24, pal.grid1, pal.grid2);
      this.grid.position.y = -0.35;
      this.scene.add(this.grid);
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  async _fetchJsonTimeout(url, ms = 3000) {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), ms);
    try {
      const r = await fetch(url, { cache: "no-store", signal: ctrl.signal });
      if (!r.ok) return null;
      return await r.json();
    } catch (_) {
      return null;
    } finally {
      clearTimeout(tid);
    }
  }

  async _adminPost(path, body) {
    if (this.apiBase === null) return;
    const tok = (this.$adminTok.value || "").trim();
    if (!tok) {
      this.$adminMsg.textContent = "Enter admin token first.";
      return;
    }
    try {
      const r = await fetch(`${this.apiBase}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${tok}`,
        },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      this.$adminMsg.textContent = r.ok ? JSON.stringify(j) : `HTTP ${r.status}: ${j.error || "failed"}`;
      this._refreshLiveStatus();
    } catch (e) {
      this.$adminMsg.textContent = e.message;
    }
  }

  async _refreshLiveStatus() {
    if (this.apiBase === null) return;
    const [h, rr] = await Promise.all([
      this._fetchJsonTimeout(`${this.apiBase}/api/health`, 3000),
      this._fetchJsonTimeout(`${this.apiBase}/api/runs`, 3000),
    ]);
    if (h) this.healthSnap = h;
    if (rr && Array.isArray(rr.runs)) {
      this.runsMeta = rr.runs;
      this._renderLangRibbon();
    }
  }

  _renderLangRibbon() {
    const items = this.runsMeta.length
      ? this.runsMeta
      : Object.keys(this.eventsByLang).map((l) => ({ lang: l, done: true, active: false, events: (this.eventsByLang[l] || []).length }));
    const modeNote = this.apiBase !== null
      ? (this.healthSnap?.running ? `LIVE · running ${(this.healthSnap.active_lang || "").toUpperCase()}` : "LIVE SPACE READY")
      : "STATIC REPLAY";
    const doneCnt = items.filter((x) => x.done).length;
    this.$langRibbon.innerHTML = [
      `<span style="font-weight:700;margin-right:4px">MULTI13 (${doneCnt}/${items.length} · ${esc(modeNote)}):</span>`,
      ...items.map((r) => {
        const cls = [
          "lang-pill",
          r.done ? "done" : "",
          r.active ? "live" : "",
          r.lang === this.selectedLang ? "selected" : "",
        ].filter(Boolean).join(" ");
        const icon = r.active ? "●" : r.done ? "✓" : "○";
        return `<button type="button" class="${cls}" data-pick-lang="${esc(r.lang)}">${esc(r.lang.toUpperCase())} ${icon}</button>`;
      }),
    ].join("");
    this.$langRibbon.querySelectorAll("[data-pick-lang]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const lang = btn.dataset.pickLang;
        this.$lang.value = lang;
        this._switchLang(lang);
      });
    });
  }

  async _tryLiveBase(base, timeoutMs = 3000) {
    const h = await this._fetchJsonTimeout(`${base}/api/health`, timeoutMs);
    if (!h || h.status !== "ok") return false;
    const rr = await this._fetchJsonTimeout(`${base}/api/runs`, timeoutMs);
    if (!rr || !Array.isArray(rr.runs) || rr.runs.length === 0) return false;
    this.apiBase = base;
    this.liveAvailable = true;
    this.healthSnap = h;
    this.runsMeta = rr.runs;
    this.$live.style.display = "inline-block";
    this.$adminBtn.style.display = "inline-block";
    if (this.$connectLive) this.$connectLive.style.display = "none";
    for (const run of rr.runs) {
      this.eventsByLang[run.lang] = [];
    }
    const active = rr.runs.find((r) => r.active);
    const firstWithEvents = rr.runs.find((r) => Number(r.events) > 0);
    this.selectedLang = String((active || firstWithEvents || rr.runs[0]).lang);
    this.followLive = true;
    this.$live.classList.add("active");
    this._populateLangSelect(rr.runs.map((x) => x.lang));
    this._renderLangRibbon();
    this._connectSse(this.selectedLang);
    if (this._statusTimer) clearInterval(this._statusTimer);
    this._statusTimer = setInterval(() => this._refreshLiveStatus(), 15000);
    return true;
  }

  async _onConnectLiveClick() {
    if (!this.liveOrigin || !this.$connectLive) return;
    this.$connectLive.disabled = true;
    this.$connectLive.textContent = "Waking Cloud Run...";
    const ok = await this._tryLiveBase(this.liveOrigin, 15000);
    if (!ok) {
      this.$connectLive.disabled = false;
      this.$connectLive.textContent = "Retry Connect Live";
    }
  }

  async _loadDataSource() {
    const wantLive = new URLSearchParams(window.location.search).get("live") === "1";
    const isSameLiveOrigin = Boolean(this.liveOrigin && window.location.origin.startsWith(this.liveOrigin));
    const isLocalHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    const candidates = [];
    if (isSameLiveOrigin || isLocalHost) {
      candidates.push({ base: "", timeout: 3000 });
    }
    if (wantLive && this.liveOrigin && !isSameLiveOrigin) {
      candidates.push({ base: this.liveOrigin, timeout: 15000 });
    }

    for (const c of candidates) {
      if (await this._tryLiveBase(c.base, c.timeout)) {
        return;
      }
    }

    if (this.liveOrigin && this.$connectLive && !isSameLiveOrigin) {
      this.$connectLive.style.display = "inline-block";
    }

    try {
      const r = await fetch(`${this.dataPrefix}/edition2_events.json`, { cache: "no-store" });
      if (r.ok) {
        const j = await r.json();
        this.eventsByLang = j.events_by_lang || {};
        const langs = j.langs || Object.keys(this.eventsByLang);
        this._populateLangSelect(langs);
        this._renderLangRibbon();
        if (langs.length) this._switchLang(langs[0]);
      }
    } catch (err) {
      this.$feed.innerHTML = `<div class="ex-card"><div class="ex-body">Waiting for Edition No. 2 stream or edition2_events.json (${esc(err.message)}).</div></div>`;
    }
  }

  _populateLangSelect(langs) {
    this.$lang.innerHTML = langs
      .map((l) => `<option value="${esc(l)}" ${l === this.selectedLang ? "selected" : ""}>${esc(l.toUpperCase())}</option>`)
      .join("");
  }

  _connectSse(lang) {
    if (this.sse) {
      this.sse.close();
      this.sse = null;
    }
    this.eventsByLang[lang] = [];
    this.seekTo(0);
    const sse = new EventSource(`${this.apiBase}/api/stream?lang=${encodeURIComponent(lang)}`);
    sse.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        (this.eventsByLang[lang] = this.eventsByLang[lang] || []).push(ev);
        this._updateScrubberMax();
        if (this.followLive) {
          this.seekTo(this.eventsByLang[lang].length);
          this._spawnPulse(ev);
        }
      } catch (_) {}
    };
    this.sse = sse;
  }

  _switchLang(lang) {
    this.selectedLang = lang;
    this._renderLangRibbon();
    if (this.followLive && this.liveAvailable) {
      this._connectSse(lang);
      return;
    }
    this.seekTo(0);
    this._updateScrubberMax();
  }

  _updateScrubberMax() {
    const evs = this.eventsByLang[this.selectedLang] || [];
    this.$scrub.max = String(evs.length);
    this.$scrub.value = String(this.cursor);
    this.$stepLbl.textContent = `${this.cursor} / ${evs.length}`;
  }

  focusCluster(slug) {
    this.selectedCluster = slug;
    this.$cluster.value = slug;
    if (this.useThree) {
      if (slug === "ALL") {
        this.camTarget = { x: 0, y: 0, z: 0 };
        this.camDesiredPos = { x: 0, y: 18, z: 22 };
      } else if (this.clusterGroups[slug]) {
        const c = this.clusterGroups[slug].center;
        this.camTarget = { x: c.x, y: 0, z: c.z };
        this.camDesiredPos = { x: c.x * 1.15, y: 7.5, z: c.z + 8.5 };
      }
    }
    this._renderHud();
  }

  seekTo(targetIdx) {
    const evs = this.eventsByLang[this.selectedLang] || [];
    const clamped = Math.max(0, Math.min(evs.length, targetIdx));
    if (clamped < this.cursor) {
      this.cursor = 0;
      this.clusterStates = {};
      this._resetClusterMeshes();
      if (this.chainGroup) this.chainGroup.clear();
      this.$feed.innerHTML = "";
    }
    while (this.cursor < clamped) {
      const ev = evs[this.cursor];
      this._applyEvent(ev, clamped - this.cursor <= 3);
      this.cursor += 1;
    }
    this._updateScrubberMax();
    this._renderHud();
  }

  _resetClusterMeshes() {
    for (const slug of CLUSTER_SLUGS) {
      const cg = this.clusterGroups?.[slug];
      if (cg) cg.dynGroup.clear();
      const lbl = this.labelNodes.find((l) => l.clusterSlug === slug);
      if (lbl) lbl.el.textContent = lbl.baseText;
    }
    this.agentPositions = {};
  }

  _applyEvent(ev, animatePulse) {
    const cl = ev.cluster;
    if (cl && cl !== "coordinator") {
      if (!this.clusterStates[cl]) {
        this.clusterStates[cl] = { n_agents: 0, topology: "-", round: 0, exchanges: 0, shadowDelta: "-", nativeParse: "-", unanimous: false };
      }
      const st = this.clusterStates[cl];
      st.round = Math.max(st.round, ev.round_num || 0);
      if (ev.kind === "cluster_spawn") {
        st.n_agents = ev.meta?.n_agents || 3;
        st.topology = ev.meta?.topology || "star";
        this._buildClusterAgentsMesh(cl, st.n_agents, st.topology, ev.meta?.edges || []);
      } else if (ev.kind === "agent_exchange" || ev.kind === "consensus_vote") {
        st.exchanges += 1;
      } else if (ev.kind === "consensus_reached") {
        st.unanimous = true;
      } else if (ev.kind === "shadow_patch") {
        st.shadowDelta = ev.label || "-";
      } else if (ev.kind === "native_generation") {
        st.nativeParse = ev.label || "-";
      }
    }
    this._appendChainBlock(ev);
    if (animatePulse) this._spawnPulse(ev);
    this._appendFeedCard(ev);
  }

  _buildClusterAgentsMesh(slug, nAgents, topology, edges) {
    const cg = this.clusterGroups?.[slug];
    if (!cg || !this.THREE) return;
    const THREE = this.THREE;
    const pal = getThemePalette();
    cg.dynGroup.clear();
    const offsets = topologyOffsets(nAgents, topology);
    const localPts = offsets.map((o) => new THREE.Vector3(o[0], o[1] + 0.35, o[2]));

    localPts.forEach((pt, idx) => {
      const orb = new THREE.Mesh(
        new THREE.SphereGeometry(0.28, 12, 8),
        new THREE.MeshLambertMaterial({ color: idx === 0 ? 0x1b4d6e : 0x554f47 })
      );
      orb.position.copy(pt);
      cg.dynGroup.add(orb);
      const worldPt = pt.clone().add(cg.center);
      this.agentPositions[`${slug}:a${idx + 1}`] = worldPt;
    });
    this.agentPositions[`${slug}:all`] = cg.center.clone().add(new THREE.Vector3(0, 0.4, 0));

    const linePts = [];
    for (const [u, v] of edges) {
      const ui = Number(String(u).replace(/\D/g, "")) - 1;
      const vi = Number(String(v).replace(/\D/g, "")) - 1;
      if (localPts[ui] && localPts[vi]) {
        linePts.push(localPts[ui], localPts[vi]);
      }
    }
    if (linePts.length) {
      const geom = new THREE.BufferGeometry().setFromPoints(linePts);
      cg.dynGroup.add(new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: pal.edge })));
    }
    const lbl = this.labelNodes.find((l) => l.clusterSlug === slug);
    if (lbl) lbl.el.innerHTML = `${esc(lbl.baseText)} <small>${nAgents}·${esc(topology)}</small>`;
  }

  _appendChainBlock(ev) {
    if (!this.chainGroup || !this.THREE) return;
    const THREE = this.THREE;
    const idx = this.chainGroup.children.length;
    if (idx > 140) return;
    const col = KIND_HEX[ev.kind] || 0x1b4d6e;
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(0.24, 0.14, 0.24),
      new THREE.MeshLambertMaterial({ color: col })
    );
    const row = idx % 45;
    const lane = Math.floor(idx / 45);
    box.position.set(-14.8 + lane * 0.45, -0.22, -10.5 + row * 0.45);
    this.chainGroup.add(box);
  }

  _resolvePos(key) {
    if (this.agentPositions[key]) return this.agentPositions[key];
    if (this.anchorPositions?.[key]) return this.anchorPositions[key];
    const baseSlug = String(key || "").split(":")[0];
    if (this.anchorPositions?.[baseSlug]) return this.anchorPositions[baseSlug];
    return this.anchorPositions?.coordinator;
  }

  _spawnPulse(ev) {
    if (!this.useThree) return;
    const p1 = this._resolvePos(ev.src);
    const p2 = this._resolvePos(ev.dst);
    if (!p1 || !p2) return;
    const mesh = new this.THREE.Mesh(
      new this.THREE.SphereGeometry(0.22, 10, 8),
      new this.THREE.MeshBasicMaterial({ color: KIND_HEX[ev.kind] || 0x1b4d6e })
    );
    mesh.position.copy(p1);
    this.scene.add(mesh);
    this.pulses.push({ mesh, p1: p1.clone(), p2: p2.clone(), t: 0 });
  }

  _appendFeedCard(ev) {
    if (this.selectedCluster !== "ALL" && ev.cluster !== this.selectedCluster && ev.cluster !== "coordinator") {
      return;
    }
    const card = document.createElement("div");
    card.className = `ex-card kind-${ev.kind}`;
    card.innerHTML = `
      <div class="ex-meta">
        <span>#${ev.seq} · ${esc(SHORT_NAMES[ev.cluster] || ev.cluster)} · ${esc(ev.kind)}</span>
        <span>${esc(ev.src.split(":").pop())} &rarr; ${esc(ev.dst.split(":").pop())}</span>
      </div>
      <div class="ex-body">${esc(ev.summary)}</div>
    `;
    this.$feed.appendChild(card);
    while (this.$feed.children.length > 90) {
      this.$feed.removeChild(this.$feed.firstChild);
    }
    this.$feed.scrollTop = this.$feed.scrollHeight;
  }

  _renderHud() {
    const focus = this.selectedCluster;
    this.root.querySelector("#hkFocus").textContent = focus === "ALL" ? "ALL (9)" : (SHORT_NAMES[focus] || focus);
    const st = this.clusterStates[focus];
    if (st) {
      this.root.querySelector("#hkTopo").textContent = `${st.n_agents}x ${st.topology}`;
      this.root.querySelector("#hkRound").textContent = `R${st.round}${st.unanimous ? " ✓" : ""}`;
      this.root.querySelector("#hkEx").textContent = String(st.exchanges);
      this.root.querySelector("#hkShadow").textContent = st.shadowDelta;
      this.root.querySelector("#hkNative").textContent = st.nativeParse;
    } else {
      const totalEx = Object.values(this.clusterStates).reduce((s, x) => s + x.exchanges, 0);
      const activeC = Object.keys(this.clusterStates).length;
      this.root.querySelector("#hkTopo").textContent = `${activeC}/9 active`;
      this.root.querySelector("#hkRound").textContent = `${this.cursor} evts`;
      this.root.querySelector("#hkEx").textContent = String(totalEx);
      this.root.querySelector("#hkShadow").textContent = "-";
      this.root.querySelector("#hkNative").textContent = "-";
    }

    this.$strip.innerHTML = [
      `<button type="button" class="cluster-chip ${focus === "ALL" ? "active" : ""}" data-slug="ALL">All</button>`,
      ...CLUSTER_SLUGS.map((s) => {
        const cs = this.clusterStates[s];
        const tag = cs ? `${cs.n_agents}·${cs.topology[0].toUpperCase()}` : "";
        return `<button type="button" class="cluster-chip ${focus === s ? "active" : ""}" data-slug="${esc(s)}">${esc(SHORT_NAMES[s])}${tag ? " " + tag : ""}</button>`;
      }),
    ].join("");
    this.$strip.querySelectorAll(".cluster-chip").forEach((b) => {
      b.addEventListener("click", () => {
        this.$feed.innerHTML = "";
        this.focusCluster(b.dataset.slug);
        const evs = (this.eventsByLang[this.selectedLang] || []).slice(0, this.cursor);
        evs.slice(-60).forEach((e) => this._appendFeedCard(e));
      });
    });

    this.labelNodes.forEach((l) => {
      if (l.clusterSlug) l.el.classList.toggle("selected", l.clusterSlug === focus);
    });
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const now = performance.now();
    const dt = Math.min(0.05, (now - this._lastStepTs) / 1000);
    this._lastStepTs = now;

    if (this.playing && !this.followLive) {
      const evs = this.eventsByLang[this.selectedLang] || [];
      this._stepAcc = (this._stepAcc || 0) + dt * this.speed * 2.5;
      if (this._stepAcc >= 1 && this.cursor < evs.length) {
        const adv = Math.floor(this._stepAcc);
        this._stepAcc -= adv;
        this.seekTo(this.cursor + adv);
      }
    }

    if (this.useThree) {
      this.controls.target.lerp(new this.THREE.Vector3(this.camTarget.x, this.camTarget.y, this.camTarget.z), 0.08);
      for (let i = this.pulses.length - 1; i >= 0; i--) {
        const p = this.pulses[i];
        p.t += dt * 1.9;
        if (p.t >= 1) {
          this.scene.remove(p.mesh);
          this.pulses.splice(i, 1);
        } else {
          p.mesh.position.lerpVectors(p.p1, p.p2, p.t);
          p.mesh.position.y += Math.sin(p.t * Math.PI) * 1.65;
        }
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);

      const w = this.$stageBox.clientWidth;
      const h = this.$stageBox.clientHeight;
      for (const lbl of this.labelNodes) {
        const v = lbl.vec3.clone().project(this.camera);
        if (v.z > 1) {
          lbl.el.style.display = "none";
        } else {
          lbl.el.style.display = "block";
          lbl.el.style.left = `${((v.x * 0.5 + 0.5) * w).toFixed(1)}px`;
          lbl.el.style.top = `${((-v.y * 0.5 + 0.5) * h).toFixed(1)}px`;
        }
      }
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const el = document.getElementById("swarmTheaterMount");
  if (el) {
    new SwarmTheater(el, {
      fullscreen: el.dataset.fullscreen === "1",
      dataPrefix: el.dataset.prefix || "../../data",
      liveOrigin: el.dataset.liveOrigin || "",
    });
  }
});
