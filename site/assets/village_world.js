/**
 * Edition No. 4 — Living Lexicon Village (Three.js Traditional Sudano-Sahelian World).
 * Static-first GitHub Pages replay from ../../data/edition4*.json with opt-in live Cloud Run SSE.
 */

const VILLAGE_ZONES = {
  toguna_council: { label: "Council Toguna", x: 0.0, z: -3.5 },
  critic_bench: { label: "Argon Critic Bench", x: -4.5, z: -4.5 },
  baobab_plaza: { label: "Baobab Plaza", x: 0.0, z: 2.2 },
  market_square: { label: "Market Square (Sugu)", x: 9.5, z: 3.5 },
  banco_compounds: { label: "Banco Compounds", x: -10.5, z: 2.5 },
  artisan_forge: { label: "Numu Forge", x: 11.5, z: -6.0 },
  millet_fields: { label: "Millet Fields", x: -11.5, z: -8.0 },
  riverbank_well: { label: "Riverbank & Well", x: 3.5, z: 12.5 },
};

const MALI13_ORDER = [
  "bam", "bmq", "boz", "dtm", "ful", "kao", "mey", "mku", "myk", "ses", "snk", "spp", "taq",
];

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function getWorldPalette() {
  const isEvening = document.documentElement.getAttribute("data-theme") === "evening";
  return isEvening
    ? {
        sky: 0x1b1615,
        fog: 0x1e1715,
        ground: 0x3b2e23,
        road: 0x4b3c2d,
        banco: 0x875438,
        thatch: 0x735b32,
        water: 0x1e4054,
        sunColor: 0xffb36b,
        sunInt: 0.72,
        hemiSky: 0x625066,
        hemiGround: 0x2a1f16,
        hemiInt: 0.68,
        lanternInt: 1.5,
      }
    : {
        sky: 0xf4ebd7,
        fog: 0xf1e5cb,
        ground: 0xd9be8c,
        road: 0xc29e68,
        banco: 0xc67c4e,
        thatch: 0xb9934c,
        water: 0x3a7c96,
        sunColor: 0xfff1cf,
        sunInt: 1.15,
        hemiSky: 0xfff8e6,
        hemiGround: 0x8c6b43,
        hemiInt: 0.92,
        lanternInt: 0.2,
      };
}

function buildArticulatedVillager(THREE, agent) {
  const group = new THREE.Group();
  const robeHex = parseInt(String(agent.color || "#8a5a18").replace("#", ""), 16) || 0x8a5a18;
  const robeMat = new THREE.MeshStandardMaterial({ color: robeHex, roughness: 0.78 });
  const skinMat = new THREE.MeshStandardMaterial({ color: 0x4b2c1a, roughness: 0.72 });
  const accentMat = new THREE.MeshStandardMaterial({
    color: agent.role === "orchestrator" ? 0xd9a441 : agent.role === "critic" ? 0xf5ecd8 : 0xe8ddc2,
    roughness: 0.65,
  });

  // Legs (pivot at hip)
  const legGeo = new THREE.CylinderGeometry(0.08, 0.07, 0.56, 8);
  const leftLegPivot = new THREE.Group();
  leftLegPivot.position.set(-0.13, 0.56, 0);
  const leftLegMesh = new THREE.Mesh(legGeo, accentMat);
  leftLegMesh.position.y = -0.28;
  leftLegPivot.add(leftLegMesh);

  const rightLegPivot = new THREE.Group();
  rightLegPivot.position.set(0.13, 0.56, 0);
  const rightLegMesh = new THREE.Mesh(legGeo, accentMat);
  rightLegMesh.position.y = -0.28;
  rightLegPivot.add(rightLegMesh);

  // Flowing boubou tunic
  const scaleMod = agent.role === "orchestrator" ? 1.14 : agent.role === "critic" ? 1.1 : 1.0;
  const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.34, 0.86, 10), robeMat);
  torso.position.y = 0.95;
  torso.castShadow = true;

  // Shoulders sash / embroidered trim
  const sash = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.25, 0.14, 10), accentMat);
  sash.position.y = 1.29;

  // Arms (pivot at shoulder)
  const armGeo = new THREE.CylinderGeometry(0.065, 0.055, 0.56, 8);
  const leftArmPivot = new THREE.Group();
  leftArmPivot.position.set(-0.31, 1.3, 0);
  const leftArmMesh = new THREE.Mesh(armGeo, robeMat);
  leftArmMesh.position.y = -0.24;
  leftArmPivot.add(leftArmMesh);

  const rightArmPivot = new THREE.Group();
  rightArmPivot.position.set(0.31, 1.3, 0);
  const rightArmMesh = new THREE.Mesh(armGeo, robeMat);
  rightArmMesh.position.y = -0.24;
  rightArmPivot.add(rightArmMesh);

  // Head + kufi / headwrap
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.2, 12, 10), skinMat);
  head.position.y = 1.56;
  head.castShadow = true;

  const hatGeo =
    agent.role === "critic"
      ? new THREE.CylinderGeometry(0.21, 0.21, 0.14, 12)
      : new THREE.CylinderGeometry(0.18, 0.21, 0.12, 10);
  const hat = new THREE.Mesh(hatGeo, accentMat);
  hat.position.y = 1.72;

  group.add(leftLegPivot, rightLegPivot, torso, sash, leftArmPivot, rightArmPivot, head, hat);

  // Orchestrator ceremonial staff for Mande
  if (agent.role === "orchestrator") {
    const staff = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.025, 1.75, 6),
      new THREE.MeshStandardMaterial({ color: 0xcfa144, roughness: 0.45 })
    );
    staff.position.set(0.05, -0.1, 0.14);
    rightArmPivot.add(staff);
  }

  // Selection ring under feet
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.38, 0.48, 20),
    new THREE.MeshBasicMaterial({ color: robeHex, side: THREE.DoubleSide, transparent: true, opacity: 0.72 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.03;
  group.add(ring);

  group.scale.setScalar(scaleMod);
  group.userData = {
    agentId: agent.agent_id,
    leftLegPivot,
    rightLegPivot,
    leftArmPivot,
    rightArmPivot,
    ring,
    walkPhase: Math.random() * 6.28,
    talkingUntil: 0,
    pathQueue: [],
    posX: agent.x,
    posZ: agent.z,
  };
  group.position.set(agent.x, 0, agent.z);
  return group;
}

class LivingVillageWorld {
  constructor(mountEl, opts = {}) {
    this.root = mountEl;
    this.dataPrefix = opts.dataPrefix || "../../data";
    this.liveOrigin = (opts.liveOrigin || "").replace(/\/$/, "");
    this.selectedLang = opts.defaultLang || "bam";
    this.ed4Summary = null;
    this.eventsByLang = {};
    this.cursor = 0;
    this.playing = true;
    this.speed = 1.8;
    this.followLive = false;
    this.sse = null;
    this.activeTab = "exchanges";
    this.selectedAgentId = "mande";
    this.camPreset = "overview";
    this.THREE = null;
    this.agentNodes = new Map();
    this.projectiles = [];
    this.pulseRings = [];
    this.lanternLights = [];
    this.liveExchanges = [];
    this.liveDecisions = [];
    this.liveEvents = [];
    this.lastBubble = "Awaiting village telemetry...";
    this._stepAcc = 0;
    this._mountDom(opts.fullscreen);
    this._init();
  }

  _mountDom(fullscreen) {
    this.root.innerHTML = `
      <div class="vw-telemetry-bar" id="vwTelemetryBar">
        <div class="vw-progress-box">
          <div class="vw-progress-head">
            <span id="vwLangHeadline">Bambara (bam) &bull; Goal: 50,000 Standardized Entries</span>
            <span id="vwProgressNums">0 / 50,000</span>
          </div>
          <div class="vw-progress-track">
            <div class="vw-progress-base" id="vwProgBase" style="width:0%"></div>
            <div class="vw-progress-added" id="vwProgAdded" style="width:0%"></div>
          </div>
        </div>
        <div class="vw-stat-card"><div class="vw-stat-lbl">Baseline \\lx</div><div class="vw-stat-val" id="vwStatBase">--</div></div>
        <div class="vw-stat-card"><div class="vw-stat-lbl">Verified +\\lx / +\\va</div><div class="vw-stat-val" id="vwStatAdded">+0 / +0</div></div>
        <div class="vw-stat-card"><div class="vw-stat-lbl">Rules +.gram</div><div class="vw-stat-val" id="vwStatGram">+0</div></div>
        <div class="vw-stat-card"><div class="vw-stat-lbl">Villagers &amp; Council</div><div class="vw-stat-val" id="vwStatCast">11 agents</div></div>
      </div>

      <div class="vw-lang-ribbon" id="vwLangRibbon">
        <span style="font-weight:700;margin-right:4px">MALI13 TARGET:</span>
        <span style="color:var(--ink-muted)">Loading language telemetry...</span>
      </div>

      <div class="vw-main-grid ${fullscreen ? "fullscreen" : ""}">
        <div>
          <div class="vw-toolbar">
            <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
              <button type="button" id="vwPlayBtn">Pause</button>
              <button type="button" id="vwReplayBtn">Replay</button>
              <select id="vwSpeedSel" aria-label="Playback speed">
                <option value="0.9">Slow (0.9x)</option>
                <option value="1.8" selected>Normal (1.8x)</option>
                <option value="3.5">Fast (3.5x)</option>
              </select>
              <select id="vwCamSel" aria-label="Camera preset">
                <option value="overview">View: Full Village</option>
                <option value="toguna">View: Council Toguna</option>
                <option value="critic">View: Gemini 4 Argon Bench</option>
                <option value="market">View: Market Square</option>
                <option value="compounds">View: Banco Compounds</option>
                <option value="follow">Follow Selected Agent</option>
              </select>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
              <span id="vwModeStatus" style="color:var(--ink-muted)">Static Replay ($0 idle)</span>
              <button type="button" id="vwLiveBtn">Connect Live (Cloud Run SSE)</button>
            </div>
          </div>

          <div class="vw-stage-box" id="vwStageBox">
            <canvas id="vwCanvas"></canvas>
            <div class="vw-labels-layer" id="vwLabelsLayer"></div>
            <div class="vw-speech-bubble" id="vwSpeechBubble">Loading Sudano-Sahelian village world...</div>
          </div>

          <div class="vw-scrub-row">
            <span>Timeline:</span>
            <input type="range" id="vwScrub" min="0" max="0" value="0" />
            <span id="vwCursorTxt">0 / 0</span>
          </div>
        </div>

        <aside class="vw-side-panel">
          <div class="vw-tabs">
            <button type="button" class="vw-tab-btn active" data-tab="exchanges">Exchange Log</button>
            <button type="button" class="vw-tab-btn" data-tab="council">Council &amp; Argon</button>
            <button type="button" class="vw-tab-btn" data-tab="cast">Cast &amp; Rules</button>
          </div>
          <div class="vw-tab-body" id="vwTabBody">Loading logs...</div>
        </aside>
      </div>
    `;

    this.$stageBox = this.root.querySelector("#vwStageBox");
    this.$canvas = this.root.querySelector("#vwCanvas");
    this.$labelsLayer = this.root.querySelector("#vwLabelsLayer");
    this.$bubble = this.root.querySelector("#vwSpeechBubble");
    this.$tabBody = this.root.querySelector("#vwTabBody");
    this.$scrub = this.root.querySelector("#vwScrub");
    this.$cursorTxt = this.root.querySelector("#vwCursorTxt");
    this.$playBtn = this.root.querySelector("#vwPlayBtn");
    this.$replayBtn = this.root.querySelector("#vwReplayBtn");
    this.$speedSel = this.root.querySelector("#vwSpeedSel");
    this.$camSel = this.root.querySelector("#vwCamSel");
    this.$liveBtn = this.root.querySelector("#vwLiveBtn");
    this.$modeStatus = this.root.querySelector("#vwModeStatus");
    this.$langRibbon = this.root.querySelector("#vwLangRibbon");

    this.$playBtn.addEventListener("click", () => {
      this.playing = !this.playing;
      this.$playBtn.textContent = this.playing ? "Pause" : "Play";
    });
    this.$replayBtn.addEventListener("click", () => {
      this._disconnectSse();
      this.seekTo(0);
      this.playing = true;
      this.$playBtn.textContent = "Pause";
    });
    this.$speedSel.addEventListener("change", () => {
      this.speed = Number(this.$speedSel.value) || 1.8;
    });
    this.$camSel.addEventListener("change", () => {
      this.camPreset = this.$camSel.value;
      this._applyCameraPreset();
    });
    this.$scrub.addEventListener("input", () => {
      this.playing = false;
      this.$playBtn.textContent = "Play";
      this.seekTo(Number(this.$scrub.value));
    });
    this.$liveBtn.addEventListener("click", () => {
      if (this.followLive) this._disconnectSse();
      else this._connectLiveSse();
    });
    this.root.querySelectorAll(".vw-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.root.querySelectorAll(".vw-tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.activeTab = btn.getAttribute("data-tab") || "exchanges";
        this._renderSideTab();
      });
    });
  }

  async _init() {
    await this._initThreeScene();
    await this._loadStaticReplay();
    this._bindThemeObserver();
    this._bindCanvasClick();
    if (new URLSearchParams(window.location.search).get("live") === "1") {
      this._connectLiveSse();
    }
    this._lastFrame = performance.now();
    this._animate();
  }

  async _initThreeScene() {
    try {
      const THREE = await import("https://esm.sh/three@0.170.0");
      const { OrbitControls } = await import("https://esm.sh/three@0.170.0/examples/jsm/controls/OrbitControls.js");
      this.THREE = THREE;
      const pal = getWorldPalette();

      const renderer = new THREE.WebGLRenderer({ canvas: this.$canvas, antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(this.$stageBox.clientWidth, this.$stageBox.clientHeight, false);
      renderer.shadowMap.enabled = true;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(pal.sky);
      scene.fog = new THREE.FogExp2(pal.fog, 0.015);

      const camera = new THREE.PerspectiveCamera(45, this.$stageBox.clientWidth / this.$stageBox.clientHeight, 0.1, 160);
      camera.position.set(16, 17, 23);

      const controls = new OrbitControls(camera, this.$canvas);
      controls.enableDamping = true;
      controls.maxPolarAngle = Math.PI / 2.1;
      controls.minDistance = 8;
      controls.maxDistance = 48;

      const hemi = new THREE.HemisphereLight(pal.hemiSky, pal.hemiGround, pal.hemiInt);
      const sun = new THREE.DirectionalLight(pal.sunColor, pal.sunInt);
      sun.position.set(18, 28, 14);
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      sun.shadow.camera.left = -24;
      sun.shadow.camera.right = 24;
      sun.shadow.camera.top = 24;
      sun.shadow.camera.bottom = -24;
      scene.add(hemi, sun);

      this.renderer = renderer;
      this.scene = scene;
      this.camera = camera;
      this.controls = controls;
      this.hemiLight = hemi;
      this.sunLight = sun;

      this._buildSahelianVillage(scene, pal);

      window.addEventListener("resize", () => {
        const w = this.$stageBox.clientWidth;
        const h = this.$stageBox.clientHeight;
        if (!w || !h) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
      });
    } catch (err) {
      console.warn("Three.js WebGL unavailable:", err);
      this.THREE = null;
    }
  }

  _buildSahelianVillage(scene, pal) {
    const THREE = this.THREE;
    // Ground plane
    this.groundMat = new THREE.MeshStandardMaterial({ color: pal.ground, roughness: 0.96 });
    const ground = new THREE.Mesh(new THREE.CircleGeometry(36, 48), this.groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Compacted earth footpaths connecting Baobab Plaza -> zones
    this.roadMat = new THREE.MeshStandardMaterial({ color: pal.road, roughness: 0.94 });
    const roadPairs = [
      ["baobab_plaza", "toguna_council"],
      ["toguna_council", "critic_bench"],
      ["baobab_plaza", "market_square"],
      ["baobab_plaza", "banco_compounds"],
      ["baobab_plaza", "riverbank_well"],
      ["market_square", "artisan_forge"],
      ["banco_compounds", "millet_fields"],
    ];
    for (const [a, b] of roadPairs) {
      const za = VILLAGE_ZONES[a];
      const zb = VILLAGE_ZONES[b];
      const dx = zb.x - za.x;
      const dz = zb.z - za.z;
      const len = Math.hypot(dx, dz);
      const pathMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.45, len), this.roadMat);
      pathMesh.rotation.x = -Math.PI / 2;
      pathMesh.rotation.z = -Math.atan2(dx, dz);
      pathMesh.position.set((za.x + zb.x) / 2, 0.015, (za.z + zb.z) / 2);
      pathMesh.receiveShadow = true;
      scene.add(pathMesh);
    }

    this.bancoMat = new THREE.MeshStandardMaterial({ color: pal.banco, roughness: 0.88 });
    this.thatchMat = new THREE.MeshStandardMaterial({ color: pal.thatch, roughness: 0.92 });
    const woodMat = new THREE.MeshStandardMaterial({ color: 0x5c3c20, roughness: 0.85 });
    const leafMat = new THREE.MeshStandardMaterial({ color: 0x3b6e3c, roughness: 0.85 });

    // 1. Central Giant Baobab Tree beside Baobab Plaza (x=2.6, z=0.8)
    const baobab = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 1.35, 3.8, 12), woodMat);
    trunk.position.y = 1.9;
    trunk.castShadow = true;
    baobab.add(trunk);
    for (const [cx, cy, cz, r] of [
      [0, 4.3, 0, 2.3],
      [-1.4, 3.8, 0.8, 1.4],
      [1.5, 3.9, -0.7, 1.5],
      [0.2, 4.1, 1.3, 1.2],
    ]) {
      const crown = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 10), leafMat);
      crown.scale.set(1.15, 0.72, 1.15);
      crown.position.set(cx, cy, cz);
      crown.castShadow = true;
      baobab.add(crown);
    }
    baobab.position.set(2.9, 0, 0.5);
    scene.add(baobab);

    // 2. Central Toguna (Council Pavilion at (0, -3.5))
    const toguna = new THREE.Group();
    const plinth = new THREE.Mesh(new THREE.CylinderGeometry(3.4, 3.6, 0.25, 16), this.roadMat);
    plinth.position.y = 0.12;
    plinth.receiveShadow = true;
    toguna.add(plinth);
    for (let i = 0; i < 8; i++) {
      const ang = (i / 8) * Math.PI * 2;
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 2.0, 8), woodMat);
      post.position.set(Math.cos(ang) * 2.7, 1.1, Math.sin(ang) * 2.7);
      post.castShadow = true;
      toguna.add(post);
    }
    // Thick layered millet-stalk roof characteristic of a Malian Toguna
    const roofBase = new THREE.Mesh(new THREE.CylinderGeometry(3.55, 3.65, 0.65, 16), this.thatchMat);
    roofBase.position.y = 2.32;
    roofBase.castShadow = true;
    const roofCap = new THREE.Mesh(new THREE.ConeGeometry(3.55, 0.9, 16), this.thatchMat);
    roofCap.position.y = 3.08;
    const table = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.05, 0.55, 14), woodMat);
    table.position.y = 0.5;
    toguna.add(roofBase, roofCap, table);
    const togunaLight = new THREE.PointLight(0xffaa55, pal.lanternInt, 12);
    togunaLight.position.set(0, 1.9, 0);
    this.lanternLights.push(togunaLight);
    toguna.add(togunaLight);
    toguna.position.set(VILLAGE_ZONES.toguna_council.x, 0, VILLAGE_ZONES.toguna_council.z);
    scene.add(toguna);

    // 3. Gemini 4 Argon High Critic Bench (-4.5, -4.5)
    const criticGroup = new THREE.Group();
    const criticStone = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.28, 1.5), this.bancoMat);
    criticStone.position.y = 0.14;
    const criticPillar = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 2.2, 8), woodMat);
    criticPillar.position.set(-0.9, 1.1, -0.6);
    const banner = new THREE.Mesh(
      new THREE.PlaneGeometry(0.85, 0.55),
      new THREE.MeshStandardMaterial({ color: 0x8c1d24, side: THREE.DoubleSide })
    );
    banner.position.set(-0.48, 1.85, -0.6);
    const criticLantern = new THREE.PointLight(0xff7755, pal.lanternInt * 1.1, 9);
    criticLantern.position.set(0, 1.7, 0);
    this.lanternLights.push(criticLantern);
    criticGroup.add(criticStone, criticPillar, banner, criticLantern);
    criticGroup.position.set(VILLAGE_ZONES.critic_bench.x, 0, VILLAGE_ZONES.critic_bench.z);
    scene.add(criticGroup);

    // 4. Banco Mudbrick Compounds & Conical Granaries (western cluster around (-10.5, 2.5))
    const buildCompound = (x, z, rotY = 0) => {
      const comp = new THREE.Group();
      const house = new THREE.Mesh(new THREE.BoxGeometry(2.8, 2.1, 2.4), this.bancoMat);
      house.position.y = 1.05;
      house.castShadow = true;
      house.receiveShadow = true;
      comp.add(house);
      // Protruding wooden toron beams
      for (const bx of [-0.9, 0, 0.9]) {
        const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.85, 6), woodMat);
        beam.rotation.x = Math.PI / 2;
        beam.position.set(bx, 1.78, 0);
        comp.add(beam);
      }
      // Conical thatched granary beside the house
      const granBody = new THREE.Mesh(new THREE.CylinderGeometry(0.65, 0.72, 1.45, 10), this.bancoMat);
      granBody.position.set(1.95, 0.72, 0.45);
      granBody.castShadow = true;
      const granCap = new THREE.Mesh(new THREE.ConeGeometry(0.85, 0.9, 10), this.thatchMat);
      granCap.position.set(1.95, 1.85, 0.45);
      comp.add(granBody, granCap);
      comp.position.set(x, 0, z);
      comp.rotation.y = rotY;
      scene.add(comp);
    };
    buildCompound(-11.5, 1.2, 0.35);
    buildCompound(-13.8, 5.2, -0.4);
    buildCompound(-8.2, 5.8, 1.1);

    // 5. Market Square (Sugu) shade canopies & clay jars around (9.5, 3.5)
    const canopyColors = [0x264b6e, 0x9c4623, 0xb8862d];
    [
      [8.2, 2.2, 0],
      [11.0, 2.6, 1],
      [9.4, 5.6, 2],
    ].forEach(([sx, sz, ci]) => {
      const stall = new THREE.Group();
      for (const [px, pz] of [[-0.9, -0.7], [0.9, -0.7], [-0.9, 0.7], [0.9, 0.7]]) {
        const p = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.65, 6), woodMat);
        p.position.set(px, 0.82, pz);
        stall.add(p);
      }
      const tarp = new THREE.Mesh(
        new THREE.BoxGeometry(2.1, 0.08, 1.7),
        new THREE.MeshStandardMaterial({ color: canopyColors[ci], roughness: 0.75 })
      );
      tarp.position.y = 1.65;
      tarp.castShadow = true;
      const jar = new THREE.Mesh(
        new THREE.SphereGeometry(0.28, 10, 8),
        new THREE.MeshStandardMaterial({ color: 0xa44d2a, roughness: 0.85 })
      );
      jar.position.set(0.45, 0.25, 0.35);
      stall.add(tarp, jar);
      stall.position.set(sx, 0, sz);
      scene.add(stall);
    });

    // 6. Numu Artisan Forge (11.5, -6.0)
    const forge = new THREE.Group();
    const kiln = new THREE.Mesh(new THREE.CylinderGeometry(0.65, 0.95, 1.5, 10), this.bancoMat);
    kiln.position.y = 0.75;
    const ember = new THREE.PointLight(0xff5a1f, 1.35, 8);
    ember.position.set(0, 1.1, 0.65);
    this.lanternLights.push(ember);
    forge.add(kiln, ember);
    forge.position.set(VILLAGE_ZONES.artisan_forge.x, 0, VILLAGE_ZONES.artisan_forge.z);
    scene.add(forge);

    // 7. Millet & Sorghum Fields (-11.5, -8.0)
    const stalkMat = new THREE.MeshStandardMaterial({ color: 0x7c8f3c, roughness: 0.9 });
    for (let gx = -15.5; gx <= -8.5; gx += 1.1) {
      for (let gz = -11.5; gz <= -5.5; gz += 1.1) {
        const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 1.05, 5), stalkMat);
        stalk.position.set(gx, 0.52, gz);
        scene.add(stalk);
      }
    }

    // 8. Niger Tributary River & Stone Well (z ~ 14 / x ~ 3.5)
    this.waterMat = new THREE.MeshStandardMaterial({ color: pal.water, roughness: 0.22, metalness: 0.15 });
    const river = new THREE.Mesh(new THREE.PlaneGeometry(66, 6.5), this.waterMat);
    river.rotation.x = -Math.PI / 2;
    river.position.set(0, 0.04, 17.2);
    scene.add(river);
    const wellRim = new THREE.Mesh(new THREE.TorusGeometry(0.75, 0.18, 8, 16), this.bancoMat);
    wellRim.rotation.x = Math.PI / 2;
    wellRim.position.set(VILLAGE_ZONES.riverbank_well.x, 0.28, VILLAGE_ZONES.riverbank_well.z);
    scene.add(wellRim);
  }

  _bindThemeObserver() {
    const obs = new MutationObserver(() => {
      if (!this.THREE || !this.scene) return;
      const pal = getWorldPalette();
      this.scene.background = new this.THREE.Color(pal.sky);
      this.scene.fog.color = new this.THREE.Color(pal.fog);
      if (this.groundMat) this.groundMat.color.setHex(pal.ground);
      if (this.roadMat) this.roadMat.color.setHex(pal.road);
      if (this.bancoMat) this.bancoMat.color.setHex(pal.banco);
      if (this.thatchMat) this.thatchMat.color.setHex(pal.thatch);
      if (this.waterMat) this.waterMat.color.setHex(pal.water);
      if (this.sunLight) {
        this.sunLight.color.setHex(pal.sunColor);
        this.sunLight.intensity = pal.sunInt;
      }
      if (this.hemiLight) {
        this.hemiLight.color.setHex(pal.hemiSky);
        this.hemiLight.groundColor.setHex(pal.hemiGround);
        this.hemiLight.intensity = pal.hemiInt;
      }
      this.lanternLights.forEach((lt) => {
        lt.intensity = pal.lanternInt;
      });
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  _bindCanvasClick() {
    this.$canvas.addEventListener("click", (evt) => {
      if (!this.THREE || !this.camera) return;
      const rect = this.$canvas.getBoundingClientRect();
      const ndc = new this.THREE.Vector2(
        ((evt.clientX - rect.left) / rect.width) * 2 - 1,
        -((evt.clientY - rect.top) / rect.height) * 2 + 1
      );
      const ray = new this.THREE.Raycaster();
      ray.setFromCamera(ndc, this.camera);
      const meshes = Array.from(this.agentNodes.values());
      const hits = ray.intersectObjects(meshes, true);
      if (hits.length > 0) {
        let cur = hits[0].object;
        while (cur && !cur.userData?.agentId) cur = cur.parent;
        if (cur?.userData?.agentId) {
          this.selectAgent(cur.userData.agentId);
        }
      }
    });
  }

  selectAgent(agentId) {
    this.selectedAgentId = agentId;
    if (this.camPreset === "follow") this._applyCameraPreset();
    this._renderSideTab();
  }

  _applyCameraPreset() {
    if (!this.camera || !this.controls) return;
    if (this.camPreset === "toguna") {
      this.camera.position.set(6, 7.5, 3.5);
      this.controls.target.set(0, 1.0, -3.5);
    } else if (this.camPreset === "critic") {
      this.camera.position.set(-9, 6.5, 1.0);
      this.controls.target.set(-4.5, 1.0, -4.5);
    } else if (this.camPreset === "market") {
      this.camera.position.set(16, 8.5, 11);
      this.controls.target.set(9.5, 1.0, 3.5);
    } else if (this.camPreset === "compounds") {
      this.camera.position.set(-16, 9.0, 10.5);
      this.controls.target.set(-10.5, 1.0, 2.5);
    } else if (this.camPreset === "follow") {
      const node = this.agentNodes.get(this.selectedAgentId);
      if (node) {
        this.controls.target.set(node.position.x, 1.1, node.position.z);
        this.camera.position.set(node.position.x + 6.5, 6.2, node.position.z + 7.5);
      }
    } else {
      this.camera.position.set(16, 17, 23);
      this.controls.target.set(0, 0.6, 1);
    }
    this.controls.update();
  }

  async _loadStaticReplay() {
    try {
      const [sRes, eRes] = await Promise.all([
        fetch(`${this.dataPrefix}/edition4.json`),
        fetch(`${this.dataPrefix}/edition4_events.json`),
      ]);
      if (sRes.ok) this.ed4Summary = await sRes.json();
      if (eRes.ok) {
        const evData = await eRes.json();
        this.eventsByLang = evData.events_by_lang || {};
      }
    } catch (err) {
      console.warn("Failed loading Edition 4 static replay:", err);
    }
    this._renderLangRibbon();
    this._renderProgressTable();
    this.selectLanguage(this.selectedLang);
  }

  _renderLangRibbon() {
    const langs = (this.ed4Summary && this.ed4Summary.languages) || {};
    this.$langRibbon.innerHTML = `<span style="font-weight:700;margin-right:4px">MALI13 TARGET:</span>` +
      MALI13_ORDER.map((l) => {
        const info = langs[l] || {};
        const added = Number(info.added_lx_count || 0);
        const turns = Number(info.total_turns || 0);
        const cls = [
          "vw-lang-pill",
          l === this.selectedLang ? "active" : "",
          turns > 0 ? "has-epochs" : "",
        ].filter(Boolean).join(" ");
        return `<button type="button" class="${cls}" data-lang="${l}">${l.toUpperCase()}${added ? ` (+${added})` : ""}</button>`;
      }).join("");
    this.$langRibbon.querySelectorAll(".vw-lang-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        const lang = btn.getAttribute("data-lang");
        if (lang) this.selectLanguage(lang);
      });
    });
  }

  _renderProgressTable() {
    const tbody = document.getElementById("ed4LangProgressBody");
    if (!tbody || !this.ed4Summary) return;
    const langs = this.ed4Summary.languages || {};
    tbody.innerHTML = MALI13_ORDER.map((l) => {
      const st = langs[l] || {};
      const base = Number(st.baseline_lx_count || 0);
      const addLx = Number(st.added_lx_count || 0);
      const addVa = Number(st.added_va_count || 0);
      const addGram = Number(st.added_gram_rules_count || 0);
      const turns = Number(st.total_turns || 0);
      const pct = ((base + addLx) / 50000) * 100;
      return `<tr>
        <td><code>${l}</code></td>
        <td>${esc(st.lang_name || l)}</td>
        <td class="num">${base.toLocaleString()}</td>
        <td class="num"><strong>+${addLx}</strong></td>
        <td class="num">+${addVa}</td>
        <td class="num">+${addGram}</td>
        <td class="num">${turns}</td>
        <td class="num">${pct.toFixed(2)}%</td>
      </tr>`;
    }).join("");
  }

  selectLanguage(lang) {
    this.selectedLang = lang;
    this._renderLangRibbon();
    this.liveEvents = this.eventsByLang[lang] || [];
    const rawSt = (this.ed4Summary && this.ed4Summary.languages && this.ed4Summary.languages[lang]) || {};
    this.liveState = { ...rawSt, active_agents: (rawSt.active_agents || []).slice() };
    this._syncAgentsFromState(this.liveState);
    this.liveExchanges = (this.liveState.recent_exchanges || []).slice();
    this.liveDecisions = (this.liveState.recent_decisions || []).slice();
    this._updateTelemetryBar(this.liveState);
    this.$scrub.max = String(Math.max(0, this.liveEvents.length));
    this.seekTo(this.liveEvents.length > 0 ? Math.min(2, this.liveEvents.length) : 0);
    if (this.followLive) this._connectLiveSse();
  }

  _syncAgentsFromState(st) {
    // Clear previous 3D agent nodes cleanly
    if (this.scene) {
      for (const node of this.agentNodes.values()) this.scene.remove(node);
    }
    this.agentNodes.clear();

    const agents = (st.active_agents && st.active_agents.length > 0)
      ? st.active_agents
      : [
          { agent_id: "mande", name: "Mande", role: "orchestrator", zone: "baobab_plaza", x: 0, z: 2.2, color: "#1b4d6e", model_id: "nvidia/nemotron-3-ultra-550b-a55b" },
          { agent_id: "argon-critic", name: "Argon Critic", role: "critic", zone: "critic_bench", x: -4.5, z: -4.5, color: "#8c1d24", model_id: "gemini-4-argon" },
        ];

    // Spread agents stationed in same zone nicely in a mini-circle so they don't overlap
    const perZone = {};
    for (const ag of agents) {
      const z = ag.zone || "baobab_plaza";
      perZone[z] = perZone[z] || [];
      perZone[z].push(ag);
    }
    for (const z of Object.keys(perZone)) {
      const grp = perZone[z];
      const base = VILLAGE_ZONES[z] || { x: 0, z: 0 };
      grp.forEach((ag, idx) => {
        const ang = (idx / Math.max(1, grp.length)) * Math.PI * 2;
        const rad = grp.length > 1 ? 1.15 : 0.2;
        const px = base.x + Math.cos(ang) * rad;
        const pz = base.z + Math.sin(ang) * rad;
        const copy = { ...ag, x: px, z: pz };
        if (this.THREE && this.scene) {
          const node = buildArticulatedVillager(this.THREE, copy);
          this.scene.add(node);
          this.agentNodes.set(copy.agent_id, node);
        }
      });
    }
  }

  _updateTelemetryBar(st) {
    const base = Number(st.baseline_lx_count || 0);
    const addedLx = Number(st.added_lx_count || 0);
    const addedVa = Number(st.added_va_count || 0);
    const addedGram = Number(st.added_gram_rules_count || 0);
    const total = base + addedLx;
    const basePct = Math.min(100, (base / 50000) * 100);
    const addPct = Math.min(100 - basePct, (addedLx / 50000) * 100);

    this.root.querySelector("#vwLangHeadline").textContent =
      `${st.lang_name || this.selectedLang} (${this.selectedLang}) • Goal: 50,000 Standardized Entries`;
    this.root.querySelector("#vwProgressNums").textContent = `${total.toLocaleString()} / 50,000`;
    this.root.querySelector("#vwProgBase").style.width = `${basePct.toFixed(2)}%`;
    this.root.querySelector("#vwProgAdded").style.width = `${addPct.toFixed(2)}%`;
    this.root.querySelector("#vwStatBase").textContent = base.toLocaleString();
    this.root.querySelector("#vwStatAdded").textContent = `+${addedLx} / +${addedVa}`;
    this.root.querySelector("#vwStatGram").textContent = `+${addedGram}`;
    const nAgents = this.agentNodes.size || (st.active_agents ? st.active_agents.length : 11);
    this.root.querySelector("#vwStatCast").textContent = `${nAgents} active`;
  }

  seekTo(idx) {
    this.cursor = Math.max(0, Math.min(idx, this.liveEvents.length));
    this.$scrub.value = String(this.cursor);
    this.$cursorTxt.textContent = `${this.cursor} / ${this.liveEvents.length}`;
    if (this.cursor > 0) {
      for (let i = Math.max(0, this.cursor - 6); i < this.cursor; i++) {
        this._applyEventVisual(this.liveEvents[i], i === this.cursor - 1);
      }
    }
    this._renderSideTab();
  }

  _applyEventVisual(ev, animateFx = true) {
    if (!ev) return;
    this.lastBubble = `<strong>[Turn ${ev.turn_id}] ${esc(ev.label)}</strong><br/><span>${esc(ev.summary)}</span>`;
    this.$bubble.innerHTML = this.lastBubble;

    if (ev.kind === "agent_spawn" && ev.meta && ev.meta.agent_id && !this.agentNodes.has(ev.meta.agent_id) && this.THREE && this.scene) {
      const node = buildArticulatedVillager(this.THREE, ev.meta);
      this.scene.add(node);
      this.agentNodes.set(ev.meta.agent_id, node);
      if (ev.path && ev.path.length > 0) node.userData.pathQueue = ev.path.map((p) => ({ x: p.x, z: p.z }));
      if (animateFx) this._spawnPulseRing(ev.meta.x || -11.5, ev.meta.z || -8.0, 0x1e5c3a);
    }

    if (ev.kind === "agent_move" && ev.agent_id && this.agentNodes.has(ev.agent_id)) {
      const node = this.agentNodes.get(ev.agent_id);
      if (ev.path && ev.path.length > 0) {
        const jitter = ((ev.seq || 1) % 5 - 2) * 0.35;
        node.userData.pathQueue = ev.path.map((p, k) => ({
          x: p.x + (k === ev.path.length - 1 ? jitter : 0),
          z: p.z + (k === ev.path.length - 1 ? -jitter : 0),
        }));
      }
    }

    if (ev.kind === "dialogue_turn" && ev.agent_id && this.agentNodes.has(ev.agent_id)) {
      this.agentNodes.get(ev.agent_id).userData.talkingUntil = performance.now() + 2600;
    }

    if (animateFx && this.THREE && this.scene) {
      if (ev.kind === "courier_scroll" && ev.zone && ev.target_zone) {
        this._spawnProjectile(VILLAGE_ZONES[ev.zone], VILLAGE_ZONES[ev.target_zone], 0xd9a441);
      } else if (ev.kind === "critic_review") {
        this._spawnProjectile(VILLAGE_ZONES.toguna_council, VILLAGE_ZONES.critic_bench, 0x8c1d24);
      } else if (ev.kind === "lexicon_commit" || ev.kind === "gram_commit") {
        this._spawnPulseRing(VILLAGE_ZONES.toguna_council.x, VILLAGE_ZONES.toguna_council.z, 0x1e5c3a);
      } else if (ev.kind === "council_reject") {
        this._spawnPulseRing(VILLAGE_ZONES.toguna_council.x, VILLAGE_ZONES.toguna_council.z, 0x8c1d24);
      }
    }
  }

  _spawnProjectile(src, dst, hexColor) {
    if (!src || !dst || !this.THREE || !this.scene) return;
    const orb = new this.THREE.Mesh(
      new this.THREE.SphereGeometry(0.24, 10, 8),
      new this.THREE.MeshBasicMaterial({ color: hexColor })
    );
    this.scene.add(orb);
    this.projectiles.push({ mesh: orb, src, dst, t: 0 });
  }

  _spawnPulseRing(x, z, hexColor) {
    if (!this.THREE || !this.scene) return;
    const mesh = new this.THREE.Mesh(
      new this.THREE.RingGeometry(0.4, 0.65, 24),
      new this.THREE.MeshBasicMaterial({ color: hexColor, side: this.THREE.DoubleSide, transparent: true, opacity: 0.9 })
    );
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x, 0.08, z);
    this.scene.add(mesh);
    this.pulseRings.push({ mesh, age: 0 });
  }

  _connectLiveSse() {
    this._disconnectSse();
    if (!this.liveOrigin) return;
    this.followLive = true;
    this.$liveBtn.classList.add("active-btn");
    this.$liveBtn.textContent = "Live Cloud Run SSE Connected";
    this.$modeStatus.textContent = `Connected to Live Cloud Run (${this.selectedLang.toUpperCase()}) · Waiting for Turn 1...`;
    this.liveEvents = [];
    this.liveExchanges = [];
    this.liveDecisions = [];
    this.cursor = 0;
    this.$scrub.max = "0";
    this.$scrub.value = "0";
    this.$cursorTxt.textContent = "0 / 0";
    this.$bubble.innerHTML = `<strong>Connected to Live Cloud Run Session (${esc(this.selectedLang.toUpperCase())})</strong><br/><span>Awaiting Turn 1 events from Mande, 9 Adaption Labs agents &amp; Gemini 4 Argon...</span>`;
    this._renderSideTab();

    fetch(`${this.liveOrigin}/api/ed4/state?lang=${encodeURIComponent(this.selectedLang)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((payload) => {
        if (!payload) return;
        const st = payload.languages ? (payload.languages[this.selectedLang] || {}) : payload;
        this.liveState = { ...st, active_agents: (st.active_agents || []).slice() };
        this._syncAgentsFromState(this.liveState);
        this.liveExchanges = (st.recent_exchanges || []).slice();
        this.liveDecisions = (st.recent_decisions || []).slice();
        this._updateTelemetryBar(this.liveState);
        this._renderSideTab();
      })
      .catch(() => {});

    this.sse = new EventSource(`${this.liveOrigin}/api/ed4/stream?lang=${encodeURIComponent(this.selectedLang)}`);
    this.sse.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        if (!this.liveEvents.some((x) => x.seq === ev.seq)) {
          this.liveEvents.push(ev);
          this.$scrub.max = String(this.liveEvents.length);
          this.cursor = this.liveEvents.length;
          this.$scrub.value = String(this.cursor);
          this.$cursorTxt.textContent = `${this.cursor} / ${this.liveEvents.length}`;
          this.$modeStatus.textContent = `Live Session · Turn ${ev.turn_id} (${this.selectedLang.toUpperCase()})`;
          this._ingestLiveTelemetry(ev);
          this._applyEventVisual(ev, true);
          this._renderSideTab();
        }
      } catch (_) {}
    };
    this.sse.addEventListener("done", () => {
      this.$modeStatus.textContent = `Live Epoch Checkpointed · ${this.liveEvents.length} events`;
    });
    this.sse.onerror = () => {
      this.$modeStatus.textContent = "Live stream reconnecting...";
    };
  }

  _ingestLiveTelemetry(ev) {
    if (!this.liveState) {
      this.liveState = { lang: this.selectedLang, baseline_lx_count: 0, added_lx_count: 0, added_va_count: 0, added_gram_rules_count: 0, active_agents: [] };
    }
    const m = ev.meta || {};
    if (ev.kind === "village_init" && m.baseline_lx_count) {
      this.liveState.baseline_lx_count = Number(m.baseline_lx_count);
    } else if (ev.kind === "agent_spawn" && m.agent_id) {
      const list = this.liveState.active_agents || [];
      if (!list.some((a) => a.agent_id === m.agent_id)) {
        list.push(m);
        this.liveState.active_agents = list;
      }
    } else if (ev.kind === "lexicon_commit") {
      if (m.final_action === "COMMIT_VA") this.liveState.added_va_count = Number(this.liveState.added_va_count || 0) + 1;
      else this.liveState.added_lx_count = Number(this.liveState.added_lx_count || 0) + 1;
      if (m.candidate_raw) this.liveDecisions.push(m);
    } else if (ev.kind === "gram_commit") {
      this.liveState.added_gram_rules_count = Number(this.liveState.added_gram_rules_count || 0) + 1;
      if (m.candidate_raw) this.liveDecisions.push(m);
    } else if (ev.kind === "council_reject") {
      if (m.candidate_raw) this.liveDecisions.push(m);
    }
    this._updateTelemetryBar(this.liveState);
  }

  _disconnectSse() {
    if (this.sse) {
      this.sse.close();
      this.sse = null;
    }
    this.followLive = false;
    this.$liveBtn.classList.remove("active-btn");
    this.$liveBtn.textContent = "Connect Live (Cloud Run SSE)";
    this.$modeStatus.textContent = "Static Replay ($0 idle)";
  }

  _renderSideTab() {
    if (this.activeTab === "exchanges") {
      const visibleEvs = this.liveEvents.slice(0, this.cursor || this.liveEvents.length).slice(-28).reverse();
      if (!visibleEvs.length) {
        this.$tabBody.innerHTML = `<div class="vw-log-card">No turns recorded yet for <code>${this.selectedLang}</code>. Connect Live on Cloud Run to start an epoch.</div>`;
        return;
      }
      this.$tabBody.innerHTML = visibleEvs.map((ev) => `
        <div class="vw-log-card kind-${esc(ev.kind)}">
          <div style="display:flex;justify-content:space-between;gap:6px;font-weight:700">
            <span>#${ev.seq} &middot; ${esc(ev.label)}</span>
            <span style="color:var(--ink-muted);font-size:0.68rem">T${ev.turn_id}</span>
          </div>
          <div style="margin-top:3px;color:var(--ink-muted)">${esc(ev.summary)}</div>
        </div>
      `).join("");
      return;
    }

    if (this.activeTab === "council") {
      if (!this.liveDecisions.length) {
        this.$tabBody.innerHTML = `<div class="vw-log-card">Council ledger empty for <code>${this.selectedLang}</code>.</div>`;
        return;
      }
      this.$tabBody.innerHTML = this.liveDecisions.slice().reverse().map((d) => {
        const ok = d.final_action !== "REJECT";
        const critic = d.critic || {};
        const dx = d.dabax_check || {};
        const ortho = d.orthography_check || {};
        return `<div class="vw-log-card kind-${ok ? "lexicon_commit" : "council_reject"}">
          <div>
            <span class="vw-tag-pill ${ok ? "vw-tag-ok" : "vw-tag-reject"}">${esc(d.final_action)}</span>
            <strong>${esc(d.standardized_lx || d.candidate_raw)}</strong>
            ${d.variant_va ? `<span>(<code>\\va ${esc(d.variant_va)}</code>)</span>` : ""}
          </div>
          <div class="vw-gate-pills">
            <span class="vw-tag-pill">DNAFLA: ${ortho.is_standard ? "OK" : "FAIL"}</span>
            <span class="vw-tag-pill">DabaX: ${dx.dict_compile_ok ? "OK" : "FAIL"} (${Number(dx.regression_pct || 0).toFixed(1)}% reg)</span>
            <span class="vw-tag-pill">Argon: ${esc(critic.verdict || "N/A")}</span>
          </div>
          <div style="color:var(--ink-muted);margin-top:3px">${esc(d.summary_reason)}</div>
          ${d.mdf_block ? `<pre class="vw-mdf-pre">${esc(d.mdf_block)}</pre>` : ""}
          ${d.gram_rule ? `<pre class="vw-mdf-pre">${esc(d.gram_rule)}</pre>` : ""}
        </div>`;
      }).join("");
      return;
    }

    // Cast & Rules tab
    const st = this.liveState || (this.ed4Summary && this.ed4Summary.languages && this.ed4Summary.languages[this.selectedLang]) || {};
    const agents = st.active_agents || [];
    this.$tabBody.innerHTML = agents.map((ag) => `
      <div class="vw-log-card" style="cursor:pointer" data-pick-agent="${esc(ag.agent_id)}">
        <div style="display:flex;justify-content:space-between;font-weight:700">
          <span>${esc(ag.name)} &mdash; <small>${esc(ag.role)}</small></span>
          <code>${esc(ag.model_id)}</code>
        </div>
        <div style="color:var(--ink-muted);margin-top:2px">${esc(ag.title)} &bull; Zone: <code>${esc(ag.zone)}</code></div>
        <div style="margin-top:2px;font-size:0.72rem">${esc(ag.domain)}${ag.spawn_reason ? ` <em>(${esc(ag.spawn_reason)})</em>` : ""}</div>
      </div>
    `).join("") || `<div class="vw-log-card">Cast initializes upon first village epoch.</div>`;

    this.$tabBody.querySelectorAll("[data-pick-agent]").forEach((el) => {
      el.addEventListener("click", () => this.selectAgent(el.getAttribute("data-pick-agent")));
    });
  }

  _updateOverlayLabels() {
    if (!this.THREE || !this.camera) return;
    const w = this.$stageBox.clientWidth;
    const h = this.$stageBox.clientHeight;
    let html = "";
    for (const [zid, z] of Object.entries(VILLAGE_ZONES)) {
      const vec = new this.THREE.Vector3(z.x, 3.2, z.z).project(this.camera);
      if (vec.z > 1) continue;
      const sx = (vec.x * 0.5 + 0.5) * w;
      const sy = (-vec.y * 0.5 + 0.5) * h;
      html += `<div class="vw-zone-badge" style="left:${sx.toFixed(1)}px;top:${sy.toFixed(1)}px">${esc(z.label)}</div>`;
    }
    for (const [aid, node] of this.agentNodes.entries()) {
      const vec = new this.THREE.Vector3(node.position.x, 2.25, node.position.z).project(this.camera);
      if (vec.z > 1) continue;
      const sx = (vec.x * 0.5 + 0.5) * w;
      const sy = (-vec.y * 0.5 + 0.5) * h;
      const roleCls = aid === "mande" ? "role-orchestrator" : aid.includes("argon") ? "role-critic" : aid.startsWith("c") ? "role-council" : aid.startsWith("sp-") ? "role-specialist" : "";
      const selCls = aid === this.selectedAgentId ? "selected" : "";
      html += `<button type="button" class="vw-agent-tag ${roleCls} ${selCls}" data-lbl-agent="${esc(aid)}" style="left:${sx.toFixed(1)}px;top:${sy.toFixed(1)}px">${esc(aid.replace(/^(v\d+-|c\d+-|sp-)/, ""))}</button>`;
    }
    this.$labelsLayer.innerHTML = html;
    this.$labelsLayer.querySelectorAll("[data-lbl-agent]").forEach((b) => {
      b.addEventListener("click", () => this.selectAgent(b.getAttribute("data-lbl-agent")));
    });
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const now = performance.now();
    const dt = Math.min(0.08, (now - this._lastFrame) / 1000);
    this._lastFrame = now;

    if (this.playing && !this.followLive && this.cursor < this.liveEvents.length) {
      this._stepAcc += dt * this.speed;
      if (this._stepAcc >= 1.0) {
        this._stepAcc = 0;
        const ev = this.liveEvents[this.cursor];
        this.cursor += 1;
        this.$scrub.value = String(this.cursor);
        this.$cursorTxt.textContent = `${this.cursor} / ${this.liveEvents.length}`;
        this._applyEventVisual(ev, true);
        this._renderSideTab();
      }
    }

    // Animate walking agents along pathQueue + talking gestures
    for (const [aid, node] of this.agentNodes.entries()) {
      const ud = node.userData;
      const q = ud.pathQueue || [];
      if (q.length > 0) {
        const tgt = q[0];
        const dx = tgt.x - node.position.x;
        const dz = tgt.z - node.position.z;
        const dist = Math.hypot(dx, dz);
        if (dist < 0.14) {
          q.shift();
        } else {
          const step = Math.min(dist, dt * 4.2);
          node.position.x += (dx / dist) * step;
          node.position.z += (dz / dist) * step;
          node.rotation.y = Math.atan2(dx, dz);
          ud.walkPhase += dt * 9.5;
          const swing = Math.sin(ud.walkPhase) * 0.55;
          ud.leftLegPivot.rotation.x = swing;
          ud.rightLegPivot.rotation.x = -swing;
          ud.leftArmPivot.rotation.x = -swing * 0.7;
          ud.rightArmPivot.rotation.x = swing * 0.7;
        }
      } else {
        ud.leftLegPivot.rotation.x *= 0.85;
        ud.rightLegPivot.rotation.x *= 0.85;
        if (ud.talkingUntil > now) {
          const gest = Math.sin(now * 0.01) * 0.45;
          ud.rightArmPivot.rotation.x = -0.5 + gest;
          ud.leftArmPivot.rotation.x = -0.35 - gest * 0.5;
        } else {
          ud.leftArmPivot.rotation.x *= 0.85;
          ud.rightArmPivot.rotation.x *= 0.85;
        }
      }
      ud.ring.visible = aid === this.selectedAgentId || ud.talkingUntil > now;
    }

    // Animate flying courier scrolls
    for (let i = this.projectiles.length - 1; i >= 0; i--) {
      const p = this.projectiles[i];
      p.t += dt * 1.4;
      if (p.t >= 1) {
        this.scene.remove(p.mesh);
        this.projectiles.splice(i, 1);
      } else {
        p.mesh.position.set(
          p.src.x + (p.dst.x - p.src.x) * p.t,
          1.2 + Math.sin(p.t * Math.PI) * 2.6,
          p.src.z + (p.dst.z - p.src.z) * p.t
        );
      }
    }

    // Animate verdict pulse rings
    for (let i = this.pulseRings.length - 1; i >= 0; i--) {
      const pr = this.pulseRings[i];
      pr.age += dt;
      if (pr.age >= 1.1) {
        this.scene.remove(pr.mesh);
        this.pulseRings.splice(i, 1);
      } else {
        const s = 1 + pr.age * 4.5;
        pr.mesh.scale.set(s, s, s);
        pr.mesh.material.opacity = Math.max(0, 0.9 * (1 - pr.age / 1.1));
      }
    }

    if (this.camPreset === "follow" && this.agentNodes.has(this.selectedAgentId) && this.controls) {
      const n = this.agentNodes.get(this.selectedAgentId);
      this.controls.target.lerp(new this.THREE.Vector3(n.position.x, 1.1, n.position.z), 0.12);
    }

    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
      this._updateOverlayLabels();
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const mount = document.getElementById("villageWorldMount");
  if (!mount) return;
  new LivingVillageWorld(mount, {
    fullscreen: mount.getAttribute("data-fullscreen") === "1",
    dataPrefix: mount.getAttribute("data-prefix") || "../../data",
    liveOrigin: mount.getAttribute("data-live-origin") || "",
  });
});
