/**
 * Volume II · Edition No. 1 — Interactive Mali OSINT Map & Live Mathematical Proof Workbench.
 * Static-first GitHub Pages replay from `../../data/vol2_*.json` with opt-in Cloud Firestore
 * (`chronicles-live`) real-time `onSnapshot` / `runQuery` hydration (`Connect Live` or `?live=1`).
 */

(function () {
  "use strict";

  const FIREBASE_WEB_CONFIG = {
    apiKey: "AIzaSyCJrgRepyDcdj3hOGSirU8NWIC1EMPh9WU",
    authDomain: "mlgramv1.firebaseapp.com",
    projectId: "mlgramv1",
    storageBucket: "mlgramv1.firebasestorage.app",
    messagingSenderId: "172515745748",
    appId: "1:172515745748:web:188db7dbb811d4bbcfc041",
    databaseId: "chronicles-live",
  };
  const FS_REST_BASE =
    `https://firestore.googleapis.com/v1/projects/${FIREBASE_WEB_CONFIG.projectId}` +
    `/databases/${FIREBASE_WEB_CONFIG.databaseId}/documents`;

  function esc(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function decodeFsVal(v) {
    if (!v || typeof v !== "object") return v;
    if ("stringValue" in v) return v.stringValue;
    if ("integerValue" in v) return Number(v.integerValue);
    if ("doubleValue" in v) return Number(v.doubleValue);
    if ("booleanValue" in v) return Boolean(v.booleanValue);
    if ("nullValue" in v) return null;
    if ("arrayValue" in v) return ((v.arrayValue && v.arrayValue.values) || []).map(decodeFsVal);
    if ("mapValue" in v) return decodeFsFields((v.mapValue && v.mapValue.fields) || {});
    return null;
  }

  function decodeFsFields(fields) {
    const out = {};
    for (const [k, v] of Object.entries(fields || {})) {
      out[k] = decodeFsVal(v);
    }
    return out;
  }

  async function queryFsCollection(colName, limit) {
    const url = `${FS_REST_BASE}:runQuery`;
    const body = {
      structuredQuery: {
        from: [{ collectionId: colName }],
        limit: limit || 250,
      },
    };
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`Firestore HTTP ${resp.status}`);
    const rows = await resp.json();
    const docs = [];
    for (const r of rows) {
      if (r && r.document && r.document.fields) {
        docs.push(decodeFsFields(r.document.fields));
      }
    }
    return docs;
  }

  // ---------------------------------------------------------------------------
  // Pure JS Live Mathematical Proof Evaluator (Hawkes + Beta-Binomial + Causal)
  // ---------------------------------------------------------------------------

  function logGamma(z) {
    const g = 7;
    const p = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7,
    ];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    z -= 1;
    let x = p[0];
    for (let i = 1; i < g + 2; i++) x += p[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  function chi2Df2Pval(stat) {
    if (stat <= 0) return 1.0;
    return Math.exp(-0.5 * stat); // exact survival function for chi2(df=2)!
  }

  function evaluateHawkesInteractive(datesIso, cutoffIso, betaDecay) {
    const cutMs = Date.parse(cutoffIso || "2026-12-31");
    const pre = (datesIso || [])
      .map((d) => Date.parse(d))
      .filter((ms) => Number.isFinite(ms) && ms <= cutMs)
      .sort((a, b) => a - b);

    const post = (datesIso || [])
      .map((d) => Date.parse(d))
      .filter((ms) => Number.isFinite(ms) && ms > cutMs && ms <= cutMs + 7 * 86400000);

    const n = pre.length;
    if (n === 0) {
      return { n: 0, post7Count: post.length, mu: 0.002, alpha: 0, beta: betaDecay, eta: 0, halfLife: Math.log(2) / betaDecay, llr: 0, pVal: 1.0, p7: 0.04, curve: [] };
    }

    const t0 = pre[0];
    const T = Math.max(21, (cutMs - t0) / 86400000 + 0.5);
    const times = pre.map((ms, idx) => (ms - t0) / 86400000 + (idx % 3) * 0.05).sort((a, b) => a - b);

    const beta = Math.max(0.02, betaDecay);
    const lam0 = Math.max(n / T, 1e-5);
    const ll0 = n * Math.log(lam0) - lam0 * T;

    // 10-step closed-form EM for given user beta slider!
    let mu = Math.max(1e-5, 0.5 * lam0);
    let eta = 0.35;
    let alpha = eta * beta;

    for (let it = 0; it < 10; it++) {
      let r = 0;
      let pBg = 0;
      let pSelf = 0;
      let tail = 0;
      for (let i = 0; i < times.length; i++) {
        if (i > 0) r = Math.exp(-beta * Math.max(0, times[i] - times[i - 1])) * (1 + r);
        const lam = Math.max(1e-12, mu + alpha * r);
        pBg += mu / lam;
        pSelf += (alpha * r) / lam;
        tail += 1 - Math.exp(-beta * Math.max(0, T - times[i]));
      }
      mu = Math.max(1e-6, pBg / T);
      eta = tail > 1e-8 ? Math.min(0.92, Math.max(0, pSelf / tail)) : 0;
      alpha = eta * beta;
    }

    let r = 0;
    let ll1 = 0;
    let tailExc = 0;
    for (let i = 0; i < times.length; i++) {
      if (i > 0) r = Math.exp(-beta * Math.max(0, times[i] - times[i - 1])) * (1 + r);
      ll1 += Math.log(Math.max(1e-12, mu + alpha * r));
      tailExc += 1 - Math.exp(-beta * Math.max(0, T - times[i]));
    }
    ll1 -= mu * T + (alpha / beta) * tailExc;

    const llr = Math.max(0, ll1 - ll0);
    const pVal = n >= 3 ? chi2Df2Pval(2 * llr) : 1.0;

    let decaySum = 0;
    for (const ti of times) decaySum += Math.exp(-beta * Math.max(0, T - ti));
    const int7 = mu * 7 + (alpha / beta) * decaySum * (1 - Math.exp(-7 * beta));
    const p7 = Math.min(0.97, Math.max(0.01, 1 - Math.exp(-int7)));

    // Build SVG trajectory points across last 90d or [0, T]
    const nPts = 55;
    const curve = [];
    for (let k = 0; k <= nPts; k++) {
      const tk = (T * k) / nPts;
      let exc = 0;
      for (const ti of times) {
        if (ti <= tk) exc += Math.exp(-beta * (tk - ti));
      }
      curve.push({ t: tk, lam: mu + alpha * exc });
    }

    return {
      n,
      post7Count: post.length,
      mu,
      alpha,
      beta,
      eta,
      halfLife: Math.log(2) / beta,
      llr,
      pVal,
      p7,
      curve,
      times,
      T,
    };
  }

  function computeBetaCurve(alpha0, beta0, activeWeeks, totalWeeks) {
    const a = Math.max(0.2, alpha0 + activeWeeks);
    const b = Math.max(0.2, beta0 + Math.max(0, totalWeeks - activeWeeks));
    const mean = a / (a + b);
    const sd = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)));
    const ciLow = Math.max(0.001, mean - 1.96 * sd);
    const ciHigh = Math.min(0.999, mean + 1.96 * sd);
    const norm = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b));
    const pts = [];
    for (let i = 1; i <= 59; i++) {
      const x = i / 60;
      const y = norm * Math.pow(x, a - 1) * Math.pow(1 - x, b - 1);
      pts.push({ x, y });
    }
    return { a, b, mean, ciLow, ciHigh, pts };
  }

  function riskColor(prob) {
    if (prob >= 0.55) return "#8b1e1e";
    if (prob >= 0.35) return "#b45309";
    if (prob >= 0.18) return "#ca8a04";
    return "#2b6cb0";
  }

  // ---------------------------------------------------------------------------
  // Mount Controller
  // ---------------------------------------------------------------------------

  async function mountVol2Observatory(rootEl) {
    const prefix = rootEl.dataset.prefix || "../../data";
    const state = {
      meta: null,
      incidents: [],
      clusters: [],
      hypotheses: [],
      proofs: [],
      predictions: [],
      exchangeRecord: null,
      selectedClusterId: null,
      selectedCercleId: null,
      activeLayer: "clusters", // clusters | risk | incidents
      activeTab: "proofs", // proofs | kl | exchange | simulate | incidents
      klDim: "actor", // actor | tactic
      perspectiveFilter: "all",
      cutoffIdx: 100,
      simBeta: 0.12,
      simAlpha0: 1.0,
      simBeta0: 12.0,
      liveConnected: false,
      livePollingTimer: null,
      replayLimit: null,
      replayTimer: null,
      unsubs: [],
      allDates: [],
      cutoffIso: "2026-12-31",
    };

    rootEl.innerHTML = `
      <div class="vol2-mount">
        <div class="vol2-toolbar">
          <div class="vol2-toolbar-group">
            <button class="vol2-btn active" data-layer="clusters">Hotspots &amp; Corridors</button>
            <button class="vol2-btn" data-layer="risk">7d / 30d Predictive Hazard</button>
            <button class="vol2-btn" data-layer="incidents">Raw OSINT Incidents</button>
          </div>
          <div class="vol2-toolbar-group">
            <select class="vol2-select" id="vol2PerspFilter" title="Filter source perspective">
              <option value="all">All Perspectives (5 Tiers)</option>
              <option value="independent_event_db">UCDP Independent Event DB</option>
              <option value="official_fama_dirpa">FAMa DIRPA Official</option>
              <option value="multilateral_un">ReliefWeb / UN Agencies</option>
              <option value="sahel_press">GDELT Sahel Press</option>
              <option value="public_x_post">Public X OSINT</option>
            </select>
            <button class="vol2-btn vol2-btn-live" id="vol2LiveBtn">Connect Firestore Live</button>
          </div>
        </div>

        <div class="vol2-kpi-strip" id="vol2KpiStrip">
          <div class="vol2-kpi"><div class="k-lbl">Verified Incidents</div><div class="k-val">—</div><div class="k-sub">Loading dataset…</div></div>
          <div class="vol2-kpi"><div class="k-lbl">Clusters &amp; Hypotheses</div><div class="k-val">—</div><div class="k-sub">9 proof families + Open</div></div>
          <div class="vol2-kpi"><div class="k-lbl">Walk-Forward AUROC</div><div class="k-val">—</div><div class="k-sub">Strict t &le; T_cut</div></div>
          <div class="vol2-kpi"><div class="k-lbl">Brier / Skill (BSS)</div><div class="k-val">—</div><div class="k-sub">Calibration</div></div>
          <div class="vol2-kpi"><div class="k-lbl">Data Mode</div><div class="k-val" id="vol2ModeLbl">Static Replay</div><div class="k-sub" id="vol2UpdatedLbl">—</div></div>
        </div>

        <div class="vol2-stage">
          <div class="vol2-map-col">
            <div class="vol2-map-canvas" id="vol2MapCanvas"></div>
            <div class="vol2-scrubber">
              <label for="vol2TimeSlider">
                <span>Temporal Causal Cutoff (T<sub>cutoff</sub>)</span>
                <span id="vol2CutoffDisplay">—</span>
              </label>
              <input id="vol2TimeSlider" type="range" min="10" max="100" value="100" />
            </div>
            <div class="vol2-legend">
              <span><span class="vol2-swatch" style="background:#8b1e1e"></span>p<sub>7d</sub> &ge; 55% (Severe Surge)</span>
              <span><span class="vol2-swatch" style="background:#b45309"></span>35–54% (Elevated)</span>
              <span><span class="vol2-swatch" style="background:#ca8a04"></span>18–34% (Watch)</span>
              <span><span class="vol2-swatch" style="background:#2b6cb0"></span>&lt; 18% (Baseline)</span>
              <span>— Dashed lines: Actor diffusion corridors</span>
            </div>
          </div>

          <div class="vol2-workbench-col" id="vol2Workbench"></div>
        </div>
      </div>
    `;

    // Load initial static JSON replay
    async function loadStaticReplay() {
      const fetchJson = async (f, optional = false) => {
        const r = await fetch(`${prefix}/${f}`, { cache: "no-store" });
        if (!r.ok) {
          if (optional) return null;
          throw new Error(`${f} HTTP ${r.status}`);
        }
        return r.json();
      };
      const [meta, incidents, clusters, hypotheses, proofs, predictions, exchanges] = await Promise.all([
        fetchJson("vol2_meta.json"),
        fetchJson("vol2_incidents.json"),
        fetchJson("vol2_clusters.json"),
        fetchJson("vol2_hypotheses.json"),
        fetchJson("vol2_proofs.json"),
        fetchJson("vol2_predictions.json"),
        fetchJson("vol2_exchanges.json", true),
      ]);
      state.meta = meta;
      state.incidents = incidents || [];
      state.clusters = clusters || [];
      state.hypotheses = hypotheses || [];
      state.proofs = proofs || [];
      state.predictions = predictions || [];
      state.exchangeRecord = exchanges || (meta && meta.agent_exchanges ? { turns: meta.agent_exchanges, open_hyp_ids: [] } : null);
    }

    try {
      await loadStaticReplay();
    } catch (err) {
      rootEl.querySelector("#vol2Workbench").innerHTML =
        `<div class="vol2-proof-card"><strong>Failed to load Volume II static data:</strong> ${esc(err.message)}</div>`;
      return;
    }

    function recomputeDatesAndSelection() {
      const ds = Array.from(new Set(state.incidents.map((i) => i.date_start).filter(Boolean))).sort();
      state.allDates = ds;
      if (ds.length > 0) {
        const idx = Math.min(ds.length - 1, Math.max(0, Math.floor(((ds.length - 1) * state.cutoffIdx) / 100)));
        state.cutoffIso = ds[idx];
      }
      if (!state.selectedClusterId && state.clusters.length > 0) {
        state.selectedClusterId = state.clusters[0].cluster_id;
        state.selectedCercleId = state.clusters[0].dominant_cercle_id;
      } else if (!state.selectedCercleId && state.proofs.length > 0) {
        state.selectedCercleId = state.proofs[0].cercle_id;
      }
      const st = state.proofs.find((p) => p.cercle_id === state.selectedCercleId);
      if (st && st.hawkes_beta) {
        state.simBeta = st.hawkes_beta;
      }
    }

    recomputeDatesAndSelection();

    // Initialize Leaflet Map (or SVG fallback if offline)
    let map = null;
    let layerGroup = null;
    const mapDiv = rootEl.querySelector("#vol2MapCanvas");

    if (typeof window.L !== "undefined") {
      map = window.L.map(mapDiv, { zoomControl: true, scrollWheelZoom: true }).setView([17.15, -3.4], 5.6);

      const esriGray = window.L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        { attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ", maxZoom: 14 }
      );
      const esriRef = window.L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 14, pane: "overlayPane" }
      );
      const esriTopo = window.L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        { attribution: "Tiles &copy; Esri &mdash; Esri, Garmin, FAO, USGS", maxZoom: 14 }
      );
      const esriStreet = window.L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        { attribution: "Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, USGS", maxZoom: 14 }
      );

      const newsprintBase = window.L.layerGroup([esriGray, esriRef]).addTo(map);
      window.L.control.layers(
        {
          "Newsprint Light (Esri)": newsprintBase,
          "Topographic Relief (Esri)": esriTopo,
          "Roads & Axes (Esri Street)": esriStreet,
        },
        null,
        { position: "topright", collapsed: true }
      ).addTo(map);

      layerGroup = window.L.layerGroup().addTo(map);
    }

    function renderKpis() {
      const m = state.meta || {};
      const kpis = rootEl.querySelectorAll("#vol2KpiStrip .vol2-kpi");
      const perspCount = Object.keys(m.perspective_counts || {}).length;
      if (kpis[0]) {
        kpis[0].querySelector(".k-val").textContent = Number(m.total_verified_incidents || state.incidents.length).toLocaleString();
        kpis[0].querySelector(".k-sub").textContent = `From ${Number(m.total_raw_records || state.incidents.length).toLocaleString()} raw across ${perspCount} perspectives`;
      }
      if (kpis[1]) {
        const openCnt = state.hypotheses.filter((h) => h.origin === "agent_open").length;
        kpis[1].querySelector(".k-val").textContent = `${state.clusters.length} / ${state.hypotheses.length}`;
        kpis[1].querySelector(".k-sub").textContent = `${m.proven_or_supported_count || 0} PROVEN/SUPPORTED (${openCnt} open)`;
      }
      if (kpis[2]) {
        kpis[2].querySelector(".k-val").textContent = (m.global_auroc != null ? m.global_auroc.toFixed(3) : "—");
        kpis[2].querySelector(".k-sub").textContent = `PR-AUC ${(m.global_pr_auc != null ? m.global_pr_auc.toFixed(3) : "—")} (${m.walk_forward_folds || 0} folds)`;
      }
      if (kpis[3]) {
        kpis[3].querySelector(".k-val").textContent = `${m.global_brier != null ? m.global_brier.toFixed(3) : "—"} / ${m.global_bss != null ? m.global_bss.toFixed(2) : "—"}`;
        kpis[3].querySelector(".k-sub").textContent = "Walk-forward 7d hazard";
      }
      rootEl.querySelector("#vol2ModeLbl").textContent = state.liveConnected ? "Firestore Live" : "Static Replay";
      rootEl.querySelector("#vol2UpdatedLbl").textContent = (m.updated_at || "Snapshot").replace("T", " ").slice(0, 19);
      rootEl.querySelector("#vol2CutoffDisplay").textContent = `${state.cutoffIso} (${state.cutoffIdx}%)`;
    }

    function selectTarget(clusterId, cercleId) {
      if (clusterId) state.selectedClusterId = clusterId;
      if (cercleId) state.selectedCercleId = cercleId;
      const st = state.proofs.find((p) => p.cercle_id === state.selectedCercleId);
      if (st && st.hawkes_beta) state.simBeta = st.hawkes_beta;
      renderMap();
      renderWorkbench();
    }

    function renderMapFallbackSvg(filteredIncidents) {
      // Pure SVG projection over Mali bbox [10..25.5N, -12.5..4.5E] if Leaflet CDN unavailable
      const W = 640, H = 480;
      const proj = (lat, lon) => [
        ((lon - (-12.5)) / 17.0) * (W - 40) + 20,
        H - (((lat - 10.0) / 15.5) * (H - 40) + 20),
      ];
      let shapes = "";
      for (const e of (state.meta && state.meta.corridor_edges) || []) {
        const [x1, y1] = proj(e.from_lat, e.from_lon);
        const [x2, y2] = proj(e.to_lat, e.to_lon);
        shapes += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#6b4f3a" stroke-dasharray="4 3" stroke-width="${Math.min(4, 1 + e.weight * 0.4)}" opacity="0.55" />`;
      }
      for (const cl of state.clusters) {
        const [x, y] = proj(cl.centroid_lat, cl.centroid_lon);
        const pr = state.predictions.find((p) => p.cercle_id === cl.dominant_cercle_id);
        const col = riskColor(pr ? pr.risk_7d_prob : 0.25);
        const r = Math.min(26, 7 + Math.sqrt(cl.event_count || 1) * 1.4);
        const activeRing = cl.cluster_id === state.selectedClusterId ? 'stroke="#111" stroke-width="2.5"' : 'stroke="#fff" stroke-width="1"';
        shapes += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${col}" fill-opacity="0.72" ${activeRing}><title>${esc(cl.label)} (${cl.event_count})</title></circle>`;
      }
      for (const inc of filteredIncidents.slice(0, 120)) {
        const [x, y] = proj(inc.lat, inc.lon);
        shapes += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.2" fill="#1f2937" opacity="0.55" />`;
      }
      mapDiv.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%">${shapes}</svg>`;
    }

    function renderMap() {
      const filteredIncidents = state.incidents.filter((inc) => {
        if (state.cutoffIso && inc.date_start > state.cutoffIso) return false;
        if (state.perspectiveFilter !== "all" && !(inc.perspectives || []).includes(state.perspectiveFilter)) return false;
        return true;
      });

      if (!map || !layerGroup) {
        renderMapFallbackSvg(filteredIncidents);
        return;
      }

      layerGroup.clearLayers();
      const predByCercle = Object.fromEntries(state.predictions.map((p) => [p.cercle_id, p]));

      if (state.activeLayer === "clusters") {
        // Corridor arrows / edges
        for (const e of (state.meta && state.meta.corridor_edges) || []) {
          const line = window.L.polyline(
            [
              [e.from_lat, e.from_lon],
              [e.to_lat, e.to_lon],
            ],
            {
              color: "#5b4636",
              weight: Math.min(4.5, 1.2 + e.weight * 0.45),
              opacity: 0.55,
              dashArray: "5, 5",
            }
          );
          line.bindTooltip(`Corridor: ${esc(e.from_cercle)} &rarr; ${esc(e.to_cercle)} (${e.weight}x ${esc(e.top_actor)})`);
          line.addTo(layerGroup);
        }

        // Hotspot Cluster circles
        for (const cl of state.clusters) {
          const pr = predByCercle[cl.dominant_cercle_id];
          const p7 = pr ? pr.risk_7d_prob : 0.2;
          const isSel = cl.cluster_id === state.selectedClusterId;
          const m = window.L.circleMarker([cl.centroid_lat, cl.centroid_lon], {
            radius: Math.min(28, 8 + Math.sqrt(cl.event_count || 1) * 1.5),
            color: isSel ? "#111111" : "#ffffff",
            weight: isSel ? 3 : 1.4,
            fillColor: riskColor(p7),
            fillOpacity: 0.78,
          });
          m.bindTooltip(
            `<strong>${esc(cl.label)}</strong><br/>Events: ${cl.event_count} | p<sub>7d</sub>=${(p7 * 100).toFixed(1)}%<br/>Click to open Mathematical Proofs`
          );
          m.on("click", () => selectTarget(cl.cluster_id, cl.dominant_cercle_id));
          m.addTo(layerGroup);
        }
      }

      if (state.activeLayer === "risk") {
        for (const pr of state.predictions) {
          const isSel = pr.cercle_id === state.selectedCercleId;
          const m = window.L.circleMarker([pr.lat, pr.lon], {
            radius: 7 + pr.risk_30d_prob * 17,
            color: isSel ? "#111" : "#fff",
            weight: isSel ? 2.5 : 1,
            fillColor: riskColor(pr.risk_7d_prob),
            fillOpacity: 0.74,
          });
          m.bindTooltip(
            `<strong>${esc(pr.cercle_name)} (${esc(pr.admin1)})</strong><br/>` +
              `7d Risk: ${(pr.risk_7d_prob * 100).toFixed(1)}% | 30d Risk: ${(pr.risk_30d_prob * 100).toFixed(1)}%<br/>` +
              `Likely Actor: ${esc(pr.dominant_actor)}<br/>Next Tactic: ${esc(pr.predicted_next_tactic)}`
          );
          const matchingCl = state.clusters.find((c) => c.dominant_cercle_id === pr.cercle_id);
          m.on("click", () => selectTarget(matchingCl ? matchingCl.cluster_id : state.selectedClusterId, pr.cercle_id));
          m.addTo(layerGroup);
        }
      }

      // Overlay individual verified incidents (full if incidents tab, faint pins otherwise)
      const cap = state.activeLayer === "incidents" ? 300 : 80;
      for (const inc of filteredIncidents.slice(0, cap)) {
        const dot = window.L.circleMarker([inc.lat, inc.lon], {
          radius: state.activeLayer === "incidents" ? 5 : 2.8,
          color: "#1f2937",
          weight: state.activeLayer === "incidents" ? 1 : 0.6,
          fillColor: inc.source_count >= 2 ? "#145a32" : "#374151",
          fillOpacity: state.activeLayer === "incidents" ? 0.8 : 0.45,
        });
        dot.bindPopup(
          `<div style="font-size:12px;max-width:240px">` +
            `<strong>${esc(inc.date_start)} &bull; ${esc(inc.locality || inc.admin2)}</strong><br/>` +
            `Actor: ${esc(inc.actor_primary)}<br/>` +
            `Tactic: ${esc(inc.tactic)} | Sources (${inc.source_count}): ${esc((inc.sources || []).join(", "))}<br/>` +
            `<p style="margin:4px 0 0">${esc(inc.headline || inc.summary)}</p></div>`
        );
        dot.addTo(layerGroup);
      }
    }

    function renderHawkesSvg(hw) {
      const W = 440, H = 130, pad = 24;
      const pts = hw.curve || [];
      if (pts.length === 0) {
        return `<div class="vol2-svg-wrap"><div class="vol2-svg-title">Hawkes Intensity &lambda;(t)</div><div style="font-size:12px;opacity:.7">No events before ${esc(state.cutoffIso)}.</div></div>`;
      }
      const maxLam = Math.max(0.05, ...pts.map((p) => p.lam));
      const T = Math.max(1, hw.T || 1);
      const xy = (t, lam) => [
        pad + (t / T) * (W - 2 * pad),
        H - pad - (lam / maxLam) * (H - 2 * pad),
      ];
      const dPath = pts
        .map((p, idx) => {
          const [x, y] = xy(p.t, p.lam);
          return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
      const rug = (hw.times || [])
        .map((t) => {
          const [x] = xy(t, 0);
          return `<line x1="${x.toFixed(1)}" y1="${H - pad}" x2="${x.toFixed(1)}" y2="${H - pad + 5}" stroke="#8b1e1e" stroke-width="1.2" />`;
        })
        .join("");
      const [, yMu] = xy(0, hw.mu);
      return `
        <div class="vol2-svg-wrap">
          <div class="vol2-svg-title">Live Hawkes &lambda;(t) Trajectory vs Baseline &mu; (T<sub>cut</sub>=${esc(state.cutoffIso)})</div>
          <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
            <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#9ca3af" stroke-width="1" />
            <line x1="${pad}" y1="${yMu.toFixed(1)}" x2="${W - pad}" y2="${yMu.toFixed(1)}" stroke="#2b6cb0" stroke-dasharray="4 3" stroke-width="1.2" />
            <path d="${dPath}" fill="none" stroke="#8b1e1e" stroke-width="2" />
            ${rug}
          </svg>
        </div>`;
    }

    function renderBetaSvg(bb) {
      const W = 440, H = 120, pad = 22;
      const pts = bb.pts || [];
      const maxY = Math.max(0.1, ...pts.map((p) => p.y));
      const dPath = pts
        .map((p, idx) => {
          const x = pad + p.x * (W - 2 * pad);
          const y = H - pad - (p.y / maxY) * (H - 2 * pad);
          return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
      const xMean = pad + bb.mean * (W - 2 * pad);
      return `
        <div class="vol2-svg-wrap">
          <div class="vol2-svg-title">Beta(${bb.a.toFixed(1)}, ${bb.b.toFixed(1)}) Posterior Weekly Hazard Density &theta;</div>
          <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
            <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#9ca3af" stroke-width="1" />
            <path d="${dPath}" fill="rgba(27, 79, 114, 0.14)" stroke="#1b4f72" stroke-width="2" />
            <line x1="${xMean.toFixed(1)}" y1="10" x2="${xMean.toFixed(1)}" y2="${H - pad}" stroke="#8b1e1e" stroke-width="1.6" stroke-dasharray="3 2" />
          </svg>
        </div>`;
    }

    function renderKlDistributionSvg(bins, dimLabel, cercleName, klNats, qBh) {
      if (!bins || bins.length === 0) {
        return `<div class="vol2-svg-wrap"><div class="vol2-svg-title">Relative Entropy D<sub>KL</sub>(P<sub>c</sub> || Q<sub>-c</sub>) &bull; ${esc(dimLabel)}</div><div style="font-size:12px;opacity:.72">Insufficient located incidents (n<sub>c</sub> &ge; 8) in ${esc(cercleName)} for LOO KL estimation.</div></div>`;
      }
      const rowH = 34;
      const W = 520;
      const H = 38 + bins.length * rowH;
      const leftPad = 152;
      const barMaxW = 175;
      const contribOrigin = 398;
      const contribMaxW = 78;
      const maxProb = Math.max(0.25, ...bins.flatMap((b) => [b.p_eb || 0, b.q_loo || 0]));
      const maxAbsContrib = Math.max(0.05, ...bins.map((b) => Math.abs(b.kl_contrib || 0)));

      const rowsSvg = bins
        .map((b, idx) => {
          const y = 30 + idx * rowH;
          const pW = ((b.p_eb || 0) / maxProb) * barMaxW;
          const qW = ((b.q_loo || 0) / maxProb) * barMaxW;
          const cVal = Number(b.kl_contrib || 0);
          const cW = (Math.abs(cVal) / maxAbsContrib) * contribMaxW;
          const cX = cVal >= 0 ? contribOrigin : contribOrigin - cW;
          const cCol = cVal >= 0 ? "#8b1e1e" : "#475569";
          const label = String(b.bin || "").slice(0, 24);
          return `
            <g>
              <text x="4" y="${y + 11}" font-size="10.5" font-weight="600" fill="currentColor">${esc(label)}</text>
              <text x="4" y="${y + 23}" font-size="9" opacity="0.72" fill="currentColor">n=${b.count_c || 0} vs ${b.count_loo || 0} LOO | lift=${Number(b.log2_lift || 0).toFixed(2)}b</text>
              <rect x="${leftPad}" y="${y + 2}" width="${pW.toFixed(1)}" height="9" fill="#8b1e1e" rx="1.5" />
              <text x="${(leftPad + pW + 4).toFixed(1)}" y="${y + 10}" font-size="9" fill="#8b1e1e" font-weight="700">${((b.p_eb || 0) * 100).toFixed(1)}%</text>
              <rect x="${leftPad}" y="${y + 14}" width="${qW.toFixed(1)}" height="8" fill="#2b6cb0" opacity="0.82" rx="1.5" />
              <text x="${(leftPad + qW + 4).toFixed(1)}" y="${y + 21}" font-size="9" fill="#2b6cb0">${((b.q_loo || 0) * 100).toFixed(1)}%</text>
              <rect x="${cX.toFixed(1)}" y="${y + 6}" width="${Math.max(1.5, cW).toFixed(1)}" height="12" fill="${cCol}" opacity="0.86" rx="1.5" />
              <text x="${contribOrigin + contribMaxW + 6}" y="${y + 15}" font-size="9" fill="${cCol}" font-weight="600">${cVal >= 0 ? "+" : ""}${cVal.toFixed(3)}</text>
            </g>`;
        })
        .join("");

      return `
        <div class="vol2-svg-wrap">
          <div class="vol2-svg-title" style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px">
            <span>Relative Entropy D<sub>KL</sub><sup>EB</sup>(P<sub>${esc(cercleName)}</sub> || Q<sub>-c</sub>) &bull; ${esc(dimLabel)}</span>
            <span>D<sub>KL</sub>=${Number(klNats || 0).toFixed(4)} nats | q<sub>BH</sub>=${Number(qBh != null ? qBh : 1).toFixed(4)}</span>
          </div>
          <div class="vol2-kl-legend">
            <span><span class="vol2-swatch" style="background:#8b1e1e"></span>Local Cercle p&#770;<sup>EB</sup> (&alpha;<sub>0</sub>=K)</span>
            <span><span class="vol2-swatch" style="background:#2b6cb0"></span>LOO National q&#770;<sub>-c</sub></span>
            <span><span class="vol2-swatch" style="background:#475569"></span>Pointwise p&#770; ln(p&#770;/q&#770;)</span>
          </div>
          <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
            <text x="${leftPad}" y="16" font-size="9.5" font-weight="700" opacity="0.72" fill="currentColor">Distribution Share (p&#770; vs q&#770;)</text>
            <text x="${contribOrigin - 28}" y="16" font-size="9.5" font-weight="700" opacity="0.72" fill="currentColor">KL Contrib (nats)</text>
            <line x1="${contribOrigin}" y1="22" x2="${contribOrigin}" y2="${H - 6}" stroke="#9ca3af" stroke-dasharray="3 2" stroke-width="1" />
            ${rowsSvg}
          </svg>
        </div>`;
    }

    function renderProofCard(h) {
      const qTxt = h.q_value != null ? ` &bull; <strong>q<sub>BH</sub> =</strong> ${h.q_value}` : "";
      const openBadge = h.origin === "agent_open" ? `<span class="vol2-badge vol2-badge-open">AGENT OPEN</span>` : "";
      const provLine =
        h.proposed_by || h.challenged_by || h.adjudicated_by
          ? `<div style="font-size:0.71rem;opacity:0.78;margin-top:4px">Provenance: ${esc(h.proposed_by || "Explorer")} &rarr; ${esc(h.challenged_by || "Theorist")} &rarr; ${esc(h.adjudicated_by || "Arbiter")}</div>`
          : "";
      return `
        <div class="vol2-proof-card">
          <div class="vol2-proof-head">
            <div>${openBadge}<strong>${esc(h.title)}</strong></div>
            <span class="vol2-verdict ${esc(h.verdict)}">${esc(h.verdict)}</span>
          </div>
          <div class="vol2-formula">${esc(h.symbolic_formula)}</div>
          <div style="font-size:0.76rem;margin-bottom:5px">
            <strong>${esc(h.test_statistic_name)}:</strong> ${h.test_statistic_value} &bull;
            <strong>p =</strong> ${h.p_value}${qTxt} &bull; <strong>n =</strong> ${h.n_observations}
          </div>
          <ul style="margin:4px 0 6px 16px;padding:0;font-size:0.76rem">
            ${(h.derivation_steps || []).map((s) => `<li>${esc(s)}</li>`).join("")}
          </ul>
          <div style="font-size:0.77rem;font-style:italic">${esc(h.analyst_narrative)}</div>
          ${provLine}
        </div>`;
    }

    function renderWorkbench() {
      const wb = rootEl.querySelector("#vol2Workbench");
      const cl = state.clusters.find((c) => c.cluster_id === state.selectedClusterId) || state.clusters[0];
      const cercleId = state.selectedCercleId || (cl && cl.dominant_cercle_id);
      const proofState = state.proofs.find((p) => p.cercle_id === cercleId);
      const pred = state.predictions.find((p) => p.cercle_id === cercleId);

      const cercleEvents = state.incidents
        .filter((i) => i.cercle_id === cercleId)
        .filter((i) => state.perspectiveFilter === "all" || (i.perspectives || []).includes(state.perspectiveFilter))
        .sort((a, b) => (a.date_start < b.date_start ? 1 : -1));

      const clHypotheses = state.hypotheses
        .filter(
          (h) =>
            (cl && h.cluster_id === cl.cluster_id) ||
            h.cercle_id === cercleId ||
            h.origin === "agent_open" ||
            h.cercle_id === "mali_national"
        )
        .sort((a, b) => {
          const score = (x) =>
            (x.cercle_id === cercleId ? 4 : 0) +
            (x.origin === "agent_open" ? 2 : 0) +
            (x.verdict === "PROVEN" || x.verdict === "SUPPORTED" ? 1 : 0);
          return score(b) - score(a);
        });

      const turns =
        (state.exchangeRecord && Array.isArray(state.exchangeRecord.turns) && state.exchangeRecord.turns) ||
        (state.meta && Array.isArray(state.meta.agent_exchanges) && state.meta.agent_exchanges) ||
        [];

      const title = cl && cl.dominant_cercle_id === cercleId
        ? `${cl.label} (${pred ? pred.admin1 : cl.admin1})`
        : `${pred ? pred.cercle_name : cercleId} Sector`;

      let tabBody = "";

      if (state.activeTab === "proofs") {
        const cards = clHypotheses.slice(0, 14).map(renderProofCard).join("");
        tabBody = cards || `<div class="vol2-proof-card">Select a hotspot cluster on the map to inspect statistical proofs.</div>`;
      } else if (state.activeTab === "kl") {
        const topKlCercles = [...state.proofs]
          .filter((p) => (p.kl_actor_nats || 0) > 0 || (p.kl_tactic_nats || 0) > 0)
          .sort((a, b) => (b.kl_actor_nats + b.kl_tactic_nats) - (a.kl_actor_nats + a.kl_tactic_nats))
          .slice(0, 8);
        const pillsHtml = topKlCercles
          .map((p) => {
            const active = p.cercle_id === cercleId ? "active" : "";
            return `<button class="vol2-pill-btn ${active}" data-kl-cercle="${esc(p.cercle_id)}">${esc(p.cercle_name)} (${p.kl_actor_nats.toFixed(2)})</button>`;
          })
          .join("");
        const bins = ((proofState && proofState.kl_bins) || []).filter((b) => b.dimension === state.klDim);
        const klVal = state.klDim === "actor" ? (proofState ? proofState.kl_actor_nats : 0) : (proofState ? proofState.kl_tactic_nats : 0);
        const klQ = state.klDim === "actor" ? (proofState ? proofState.kl_actor_q_bh : 1) : (proofState ? proofState.kl_tactic_q_bh : 1);
        const jsdH = state.hypotheses.find((h) => h.hyp_id === "national_mali__H_kl_regime_jsd");
        const parH = state.hypotheses.find((h) => h.hyp_id === "national_mali__H_pareto_tail");
        const cpDate = proofState && proofState.changepoint_date ? proofState.changepoint_date : "None (BH<=0.05)";

        tabBody = `
          <div class="vol2-kl-controls">
            <div class="vol2-toolbar-group">
              <button class="vol2-btn ${state.klDim === "actor" ? "active" : ""}" data-kl-dim="actor">Actor Profile (K=5)</button>
              <button class="vol2-btn ${state.klDim === "tactic" ? "active" : ""}" data-kl-dim="tactic">Tactic Profile (K&le;8)</button>
            </div>
            <div class="vol2-kl-pills">${pillsHtml}</div>
          </div>
          ${renderKlDistributionSvg(bins, state.klDim === "actor" ? "Actor" : "Tactic", pred ? pred.cercle_name : cercleId, klVal, klQ)}
          <div class="vol2-proof-card" style="margin-top:10px">
            <strong>Cercle Information-Theoretic Telemetry (${esc(pred ? pred.cercle_name : cercleId)}):</strong><br/>
            <span style="font-size:0.78rem">
              Actor D<sub>KL</sub><sup>EB</sup>: <strong>${proofState ? proofState.kl_actor_nats.toFixed(4) : "0.0000"}</strong> nats (q<sub>BH</sub>=${proofState ? proofState.kl_actor_q_bh.toFixed(4) : "1.0000"}) &bull;
              Tactic D<sub>KL</sub><sup>EB</sup>: <strong>${proofState ? proofState.kl_tactic_nats.toFixed(4) : "0.0000"}</strong> nats (q<sub>BH</sub>=${proofState ? proofState.kl_tactic_q_bh.toFixed(4) : "1.0000"})<br/>
              Neighborhood-Inflow TE<sub>MM</sub>: <strong>${proofState ? proofState.te_inflow_bits.toFixed(4) : "0.0000"}</strong> bits (q<sub>BH</sub>=${proofState ? proofState.te_inflow_q_bh.toFixed(4) : "1.0000"}) &bull;
              Changepoint: <strong>${esc(cpDate)}</strong>
            </span>
          </div>
          ${jsdH ? renderKlDistributionSvg((jsdH.kl_bins || []).slice(0, 8), "National Regime Shift (Early vs Late)", "Mali", jsdH.test_statistic_value, jsdH.q_value) : ""}
          ${parH ? `<div class="vol2-proof-card" style="margin-top:8px"><strong>National Heavy-Tail Law:</strong> ${esc(parH.analyst_narrative)} <span class="vol2-verdict ${esc(parH.verdict)}" style="margin-left:6px">${esc(parH.verdict)}</span></div>` : ""}
        `;
      } else if (state.activeTab === "exchange") {
        const turnsAll =
          (state.exchangeRecord && Array.isArray(state.exchangeRecord.turns) && state.exchangeRecord.turns) ||
          (state.meta && Array.isArray(state.meta.agent_exchanges) && state.meta.agent_exchanges) ||
          [];
        const shownTurns = state.replayLimit != null ? turnsAll.slice(0, state.replayLimit) : turnsAll;
        const openHyps = state.hypotheses.filter((h) => h.origin === "agent_open");
        const isSweepRunning = Boolean(state.meta && state.meta.running);
        const statusPill = state.liveConnected
          ? `<span class="vol2-verdict ${isSweepRunning ? "FALSIFIED" : "PROVEN"}">${isSweepRunning ? "STREAMING LIVE TURN-BY-TURN" : "FIRESTORE LIVE SYNCED"}</span>`
          : `<span class="vol2-verdict INCONCLUSIVE">STATIC SNAPSHOT</span>`;
        const turnsHtml = shownTurns
          .map((t) => {
            const refs = Object.entries(t.math_refs || {})
              .map(([k, v]) => `<span class="vol2-badge">${esc(k)}=${esc(v)}</span>`)
              .join("");
            const roleCls = String(t.role_badge || "Explorer").toLowerCase();
            return `
              <div class="vol2-exchange-turn vol2-role-${esc(roleCls)}">
                <div class="vol2-exchange-head">
                  <div>
                    <span class="vol2-role-pill ${esc(roleCls)}">${esc(t.role_badge || "Agent")}</span>
                    <strong>#${t.turn_index || "?"} ${esc(t.agent_name || t.agent_id)}</strong>
                  </div>
                  <span class="vol2-model-tag">${esc(t.model_id || "")}</span>
                </div>
                <div class="vol2-exchange-msg">${esc(t.message)}</div>
                <div class="vol2-exchange-foot">
                  <div>${refs}</div>
                  ${t.verdict_after ? `<span class="vol2-verdict ${esc(t.verdict_after)}">${esc(t.verdict_after)}</span>` : ""}
                </div>
              </div>`;
          })
          .join("");
        const openCards = openHyps.map(renderProofCard).join("");
        tabBody = `
          <div class="vol2-proof-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
              <strong>Autonomous Agent Proof Observatory &amp; Peer Debate</strong>
              ${statusPill}
            </div>
            <div style="font-size:0.77rem;opacity:0.84;margin-top:4px">
              ${esc((state.exchangeRecord && state.exchangeRecord.topic) || "Multi-agent deduction, estimator bias challenge & BH-FDR adjudication.")}
              <br/><span>Sweep: <code>${esc((state.exchangeRecord && state.exchangeRecord.sweep_id) || (state.meta && state.meta.sweep_id) || "snapshot")}</code> &bull; Updated: <strong>${esc((state.exchangeRecord && state.exchangeRecord.updated_at) || (state.meta && state.meta.updated_at) || "—")}</strong></span>
            </div>
            <div class="vol2-toolbar-group" style="margin-top:8px">
              ${!state.liveConnected ? `<button class="vol2-btn vol2-btn-live" id="vol2ExLiveBtn">Connect Firestore Real-Time Stream</button>` : `<button class="vol2-btn" id="vol2ExRefreshBtn">Pull Latest Firestore Turns Now</button>`}
              <button class="vol2-btn" id="vol2ExReplayBtn">${state.replayTimer ? "Pause Animated Replay" : "Animate Step-by-Step Replay"}</button>
            </div>
          </div>
          <div class="vol2-exchange-feed">${turnsHtml || `<div class="vol2-proof-card">${isSweepRunning ? "Cloud Run sweep active — waiting for Turn 1 (Explorer) to stream via Firestore..." : "No agent exchange turns recorded yet."}</div>`}</div>
          ${openCards ? `<h4 style="margin:12px 0 6px;font-size:0.86rem;text-transform:uppercase;letter-spacing:0.04em">Deduced Open Hypotheses (${openHyps.length})</h4>${openCards}` : ""}
        `;
      } else if (state.activeTab === "simulate") {
        const eventDates = cercleEvents.map((e) => e.date_start);
        const hw = evaluateHawkesInteractive(eventDates, state.cutoffIso, state.simBeta);
        const activeWks = proofState ? proofState.beta_active_weeks : Math.min(52, hw.n);
        const totalWks = proofState ? proofState.beta_total_weeks : 52;
        const bb = computeBetaCurve(state.simAlpha0, state.simBeta0, activeWks, totalWks);
        const hitNext7 = hw.post7Count > 0 ? "REALIZED HIT (+7d)" : "QUIET WINDOW (+7d)";

        tabBody = `
          <div class="vol2-sliders">
            <div class="vol2-slider-item">
              <label><span>Hawkes Decay &beta;</span><span>${state.simBeta.toFixed(2)} d<sup>-1</sup></span></label>
              <input id="simBetaSlider" type="range" min="0.03" max="0.60" step="0.01" value="${state.simBeta}" />
            </div>
            <div class="vol2-slider-item">
              <label><span>Beta Prior &alpha;<sub>0</sub></span><span>${state.simAlpha0.toFixed(1)}</span></label>
              <input id="simAlphaSlider" type="range" min="0.5" max="8.0" step="0.5" value="${state.simAlpha0}" />
            </div>
            <div class="vol2-slider-item">
              <label><span>Beta Prior &beta;<sub>0</sub></span><span>${state.simBeta0.toFixed(1)}</span></label>
              <input id="simBeta0Slider" type="range" min="2.0" max="30.0" step="1.0" value="${state.simBeta0}" />
            </div>
          </div>
          <div class="vol2-proof-card" style="margin-top:8px">
            <strong>Live Recomputed Proof @ Cutoff ${esc(state.cutoffIso)}:</strong><br/>
            <span style="font-size:0.79rem">
              Events &le; T<sub>cut</sub>: <strong>${hw.n}</strong> &bull;
              Branching &eta;=&alpha;/&beta;: <strong>${hw.eta.toFixed(3)}</strong> &bull;
              t<sub>1/2</sub>: <strong>${hw.halfLife.toFixed(1)}d</strong> &bull;
              LLR 2&Delta;&#8467;: <strong>${(2 * hw.llr).toFixed(2)}</strong> (p=${hw.pVal.toFixed(4)})<br/>
              Causal 7d Hazard P<sub>causal</sub>: <strong>${(hw.p7 * 100).toFixed(1)}%</strong> &bull;
              Out-of-sample (T<sub>cut</sub>, +7d]: <strong>${hitNext7} (${hw.post7Count} ev)</strong><br/>
              Beta Posterior Mean: <strong>${(bb.mean * 100).toFixed(1)}%</strong> (95% CrI: ${(bb.ciLow * 100).toFixed(1)}%–${(bb.ciHigh * 100).toFixed(1)}%)
            </span>
          </div>
          ${renderHawkesSvg(hw)}
          ${renderBetaSvg(bb)}
        `;
      } else {
        const rows = cercleEvents.slice(0, 60).map((e) => `
          <div class="vol2-event-row">
            <div>
              <strong>${esc(e.date_start)}</strong> &bull; ${esc(e.locality || e.admin2)} &bull;
              <em>${esc(e.actor_primary)}</em>
            </div>
            <div style="margin:2px 0">
              <span class="vol2-badge">${esc(e.tactic)}</span>
              <span class="vol2-badge">${esc(e.corroboration_tier)} (${e.source_count} src)</span>
              ${e.fatalities_best ? `<span class="vol2-badge">${e.fatalities_best} fat.</span>` : ""}
            </div>
            <div style="opacity:.86">${esc(e.headline || e.summary)}</div>
          </div>
        `).join("");
        tabBody = `<div class="vol2-events-list">${rows || '<div class="vol2-event-row">No matching events in filter.</div>'}</div>`;
      }

      wb.innerHTML = `
        <div class="vol2-wb-header">
          <h3>${esc(title)}</h3>
          <div class="vol2-wb-meta">
            7d Forecast: <strong>${pred ? (pred.risk_7d_prob * 100).toFixed(1) + "%" : "—"}</strong> &bull;
            30d Forecast: <strong>${pred ? (pred.risk_30d_prob * 100).toFixed(1) + "%" : "—"}</strong> &bull;
            Predicted Next Tactic: <strong>${esc(pred ? pred.predicted_next_tactic : "—")}</strong>
          </div>
        </div>
        <div class="vol2-tabs">
          <button class="vol2-tab-btn ${state.activeTab === "proofs" ? "active" : ""}" data-tab="proofs">Mathematical Proofs (${clHypotheses.length})</button>
          <button class="vol2-tab-btn ${state.activeTab === "kl" ? "active" : ""}" data-tab="kl">Relative Entropy (KL)</button>
          <button class="vol2-tab-btn ${state.activeTab === "exchange" ? "active" : ""}" data-tab="exchange">Agent Proof Exchange (${turns.length})</button>
          <button class="vol2-tab-btn ${state.activeTab === "simulate" ? "active" : ""}" data-tab="simulate">Live Interactive Lab</button>
          <button class="vol2-tab-btn ${state.activeTab === "incidents" ? "active" : ""}" data-tab="incidents">Underlying Data (${cercleEvents.length})</button>
        </div>
        ${tabBody}
      `;

      wb.querySelectorAll("[data-tab]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.activeTab = btn.dataset.tab;
          renderWorkbench();
        });
      });

      wb.querySelectorAll("[data-kl-dim]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.klDim = btn.dataset.klDim;
          renderWorkbench();
        });
      });

      wb.querySelectorAll("[data-kl-cercle]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const cid = btn.dataset.klCercle;
          const matchingCl = state.clusters.find((c) => c.dominant_cercle_id === cid);
          selectTarget(matchingCl ? matchingCl.cluster_id : state.selectedClusterId, cid);
        });
      });

      const bS = wb.querySelector("#simBetaSlider");
      const a0S = wb.querySelector("#simAlphaSlider");
      const b0S = wb.querySelector("#simBeta0Slider");
      if (bS) {
        bS.addEventListener("input", (ev) => {
          state.simBeta = Number(ev.target.value);
          renderWorkbench();
        });
      }
      if (a0S) {
        a0S.addEventListener("input", (ev) => {
          state.simAlpha0 = Number(ev.target.value);
          renderWorkbench();
        });
      }
      if (b0S) {
        b0S.addEventListener("input", (ev) => {
          state.simBeta0 = Number(ev.target.value);
          renderWorkbench();
        });
      }
      const exLiveBtn = wb.querySelector("#vol2ExLiveBtn");
      if (exLiveBtn) {
        exLiveBtn.addEventListener("click", () => connectFirestoreLive());
      }
      const exRefBtn = wb.querySelector("#vol2ExRefreshBtn");
      if (exRefBtn) {
        exRefBtn.addEventListener("click", () => {
          if (state.liveRefreshFn) state.liveRefreshFn();
        });
      }
      const exRepBtn = wb.querySelector("#vol2ExReplayBtn");
      if (exRepBtn) {
        exRepBtn.addEventListener("click", () => {
          if (state.replayTimer) {
            clearInterval(state.replayTimer);
            state.replayTimer = null;
            state.replayLimit = null;
            renderWorkbench();
            return;
          }
          const currentTurns =
            (state.exchangeRecord && Array.isArray(state.exchangeRecord.turns) && state.exchangeRecord.turns) ||
            (state.meta && Array.isArray(state.meta.agent_exchanges) && state.meta.agent_exchanges) ||
            [];
          const total = currentTurns.length;
          if (!total) return;
          state.replayLimit = 1;
          renderWorkbench();
          state.replayTimer = setInterval(() => {
            if (state.replayLimit == null || state.replayLimit >= total) {
              clearInterval(state.replayTimer);
              state.replayTimer = null;
              state.replayLimit = null;
              renderWorkbench();
              return;
            }
            state.replayLimit += 1;
            renderWorkbench();
          }, 1400);
        });
      }
    }

    async function connectFirestoreLive() {
      const btn = rootEl.querySelector("#vol2LiveBtn");
      if (state.liveConnected) {
        for (const fn of state.unsubs) {
          try { fn(); } catch (_) {}
        }
        state.unsubs = [];
        if (state.livePollingTimer) {
          clearInterval(state.livePollingTimer);
          state.livePollingTimer = null;
        }
        state.liveConnected = false;
        btn.classList.remove("connected");
        btn.textContent = "Connect Firestore Live";
        renderKpis();
        renderWorkbench();
        return;
      }

      btn.textContent = "Connecting…";
      const fetchLiveExchangeRest = async () => {
        try {
          const r = await fetch(`${FS_REST_BASE}/vol2_exchanges/latest`, { cache: "no-store" });
          if (r.ok) {
            const j = await r.json();
            if (j && j.fields) {
              state.exchangeRecord = decodeFsFields(j.fields);
              return;
            }
          }
        } catch (_) {}
        const allEx = await queryFsCollection("vol2_exchanges", 10).catch(() => []);
        if (allEx.length) {
          allEx.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
          state.exchangeRecord = allEx[0];
        }
      };

      const fetchMetaRest = async () => {
        try {
          const r = await fetch(`${FS_REST_BASE}/vol2_meta/overview`, { cache: "no-store" });
          if (r.ok) {
            const j = await r.json();
            if (j && j.fields) state.meta = decodeFsFields(j.fields);
          }
        } catch (_) {}
      };

      try {
        const [appMod, fsMod] = await Promise.all([
          import("https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js"),
          import("https://www.gstatic.com/firebasejs/11.6.0/firebase-firestore.js"),
        ]);
        const app = appMod.getApps().length
          ? appMod.getApps()[0]
          : appMod.initializeApp(FIREBASE_WEB_CONFIG, "vol2-chronicles");
        const db = fsMod.getFirestore(app, FIREBASE_WEB_CONFIG.databaseId);

        const refreshAllRestFallback = async () => {
          const [cl, hy, prf, prd, inc] = await Promise.all([
            queryFsCollection("vol2_clusters", 50),
            queryFsCollection("vol2_hypotheses", 250),
            queryFsCollection("vol2_proofs", 50),
            queryFsCollection("vol2_predictions", 50),
            queryFsCollection("vol2_incidents", 200),
            fetchLiveExchangeRest(),
          ]);
          if (cl.length) state.clusters = cl;
          if (hy.length) state.hypotheses = hy;
          if (prf.length) state.proofs = prf;
          if (prd.length) state.predictions = prd;
          if (inc.length) state.incidents = inc;
          if ((!state.exchangeRecord || !Array.isArray(state.exchangeRecord.turns)) && state.meta && Array.isArray(state.meta.agent_exchanges)) {
            state.exchangeRecord = { turns: state.meta.agent_exchanges, open_hyp_ids: [], sweep_id: state.meta.sweep_id, updated_at: state.meta.updated_at };
          }
          recomputeDatesAndSelection();
          renderKpis();
          renderMap();
          renderWorkbench();
        };
        state.liveRefreshFn = refreshAllRestFallback;

        let lastSweepId = "";
        const unsubMeta = fsMod.onSnapshot(
          fsMod.doc(db, "vol2_meta", "overview"),
          (snap) => {
            if (snap.exists()) {
              const d = snap.data();
              state.meta = d;
              renderKpis();
              if (d && d.sweep_id && d.sweep_id !== lastSweepId && !d.running) {
                lastSweepId = d.sweep_id;
                refreshAllRestFallback().catch(() => {});
              } else {
                renderWorkbench();
              }
            }
          },
          () => {}
        );
        const unsubEx = fsMod.onSnapshot(
          fsMod.doc(db, "vol2_exchanges", "latest"),
          (snap) => {
            if (snap.exists()) {
              state.exchangeRecord = snap.data();
              renderWorkbench();
            }
          },
          () => {}
        );
        state.unsubs.push(unsubMeta, unsubEx);
        // Periodic REST sync every 6s while connected guarantees real-time updates even if Enterprise WebChannel throttles
        state.livePollingTimer = setInterval(async () => {
          if (!state.liveConnected) return;
          await Promise.all([fetchMetaRest(), fetchLiveExchangeRest()]);
          renderKpis();
          renderWorkbench();
        }, 6000);

        state.liveConnected = true;
        btn.classList.add("connected");
        btn.textContent = "Live Connected (Disconnect)";
        await refreshAllRestFallback();
      } catch (_err) {
        try {
          await Promise.all([fetchMetaRest(), fetchLiveExchangeRest()]);
          state.liveConnected = true;
          btn.classList.add("connected");
          btn.textContent = "Live Polling (REST)";
          state.livePollingTimer = setInterval(async () => {
            if (!state.liveConnected) return;
            await Promise.all([fetchMetaRest(), fetchLiveExchangeRest()]);
            renderKpis();
            renderWorkbench();
          }, 6000);
          renderKpis();
          renderWorkbench();
        } catch (err2) {
          btn.textContent = `Live Unavailable (${err2.message.slice(0, 18)})`;
        }
      }
    }

    // Wire toolbar controls
    rootEl.querySelectorAll("[data-layer]").forEach((btn) => {
      btn.addEventListener("click", () => {
        rootEl.querySelectorAll("[data-layer]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeLayer = btn.dataset.layer;
        renderMap();
      });
    });

    rootEl.querySelector("#vol2PerspFilter").addEventListener("change", (ev) => {
      state.perspectiveFilter = ev.target.value;
      renderMap();
      renderWorkbench();
    });

    rootEl.querySelector("#vol2TimeSlider").addEventListener("input", (ev) => {
      state.cutoffIdx = Number(ev.target.value);
      recomputeDatesAndSelection();
      renderKpis();
      renderMap();
      if (state.activeTab === "simulate") renderWorkbench();
    });

    rootEl.querySelector("#vol2LiveBtn").addEventListener("click", () => {
      connectFirestoreLive();
    });

    renderKpis();
    renderMap();
    renderWorkbench();

    if (new URLSearchParams(window.location.search).get("live") === "1") {
      connectFirestoreLive();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const mount = document.getElementById("vol2MaliMapMount");
    if (mount) {
      mountVol2Observatory(mount);
    }
  });
})();
