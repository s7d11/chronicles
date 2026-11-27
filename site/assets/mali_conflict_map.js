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
      selectedClusterId: null,
      selectedCercleId: null,
      activeLayer: "clusters", // clusters | risk | incidents
      activeTab: "proofs", // proofs | simulate | incidents
      perspectiveFilter: "all",
      cutoffIdx: 100,
      simBeta: 0.12,
      simAlpha0: 1.0,
      simBeta0: 12.0,
      liveConnected: false,
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
          <div class="vol2-kpi"><div class="k-lbl">Clusters &amp; Hypotheses</div><div class="k-val">—</div><div class="k-sub">5 test families</div></div>
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
      const fetchJson = async (f) => {
        const r = await fetch(`${prefix}/${f}`, { cache: "no-store" });
        if (!r.ok) throw new Error(`${f} HTTP ${r.status}`);
        return r.json();
      };
      const [meta, incidents, clusters, hypotheses, proofs, predictions] = await Promise.all([
        fetchJson("vol2_meta.json"),
        fetchJson("vol2_incidents.json"),
        fetchJson("vol2_clusters.json"),
        fetchJson("vol2_hypotheses.json"),
        fetchJson("vol2_proofs.json"),
        fetchJson("vol2_predictions.json"),
      ]);
      state.meta = meta;
      state.incidents = incidents || [];
      state.clusters = clusters || [];
      state.hypotheses = hypotheses || [];
      state.proofs = proofs || [];
      state.predictions = predictions || [];
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
      window.L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
        attribution: "&copy; OpenStreetMap &copy; CARTO | Sebeni Chronicle",
        maxZoom: 14,
      }).addTo(map);
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
        kpis[1].querySelector(".k-val").textContent = `${state.clusters.length} / ${state.hypotheses.length}`;
        kpis[1].querySelector(".k-sub").textContent = `${m.proven_or_supported_count || 0} PROVEN or SUPPORTED (a=0.05)`;
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

      const clHypotheses = state.hypotheses.filter(
        (h) => (cl && h.cluster_id === cl.cluster_id) || h.cercle_id === cercleId
      );

      const title = cl && cl.dominant_cercle_id === cercleId
        ? `${cl.label} (${pred ? pred.admin1 : cl.admin1})`
        : `${pred ? pred.cercle_name : cercleId} Sector`;

      let tabBody = "";

      if (state.activeTab === "proofs") {
        const cards = clHypotheses.slice(0, 5).map((h) => `
          <div class="vol2-proof-card">
            <div class="vol2-proof-head">
              <strong>${esc(h.title)}</strong>
              <span class="vol2-verdict ${esc(h.verdict)}">${esc(h.verdict)}</span>
            </div>
            <div class="vol2-formula">${esc(h.symbolic_formula)}</div>
            <div style="font-size:0.76rem;margin-bottom:5px">
              <strong>${esc(h.test_statistic_name)}:</strong> ${h.test_statistic_value} &bull;
              <strong>p =</strong> ${h.p_value} &bull; <strong>n =</strong> ${h.n_observations}
            </div>
            <ul style="margin:4px 0 6px 16px;padding:0;font-size:0.76rem">
              ${(h.derivation_steps || []).map((s) => `<li>${esc(s)}</li>`).join("")}
            </ul>
            <div style="font-size:0.77rem;font-style:italic">${esc(h.analyst_narrative)}</div>
          </div>
        `).join("");
        tabBody = cards || `<div class="vol2-proof-card">Select a hotspot cluster on the map to inspect its 5 statistical proofs.</div>`;
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
          <button class="vol2-tab-btn ${state.activeTab === "simulate" ? "active" : ""}" data-tab="simulate">Live Interactive Proof Lab</button>
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
    }

    async function connectFirestoreLive() {
      const btn = rootEl.querySelector("#vol2LiveBtn");
      if (state.liveConnected) {
        for (const fn of state.unsubs) {
          try { fn(); } catch (_) {}
        }
        state.unsubs = [];
        state.liveConnected = false;
        btn.classList.remove("connected");
        btn.textContent = "Connect Firestore Live";
        renderKpis();
        return;
      }

      btn.textContent = "Connecting…";
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
            queryFsCollection("vol2_hypotheses", 120),
            queryFsCollection("vol2_proofs", 50),
            queryFsCollection("vol2_predictions", 50),
            queryFsCollection("vol2_incidents", 200),
          ]);
          if (cl.length) state.clusters = cl;
          if (hy.length) state.hypotheses = hy;
          if (prf.length) state.proofs = prf;
          if (prd.length) state.predictions = prd;
          if (inc.length) state.incidents = inc;
          recomputeDatesAndSelection();
          renderKpis();
          renderMap();
          renderWorkbench();
        };

        const unsubMeta = fsMod.onSnapshot(
          fsMod.doc(db, "vol2_meta", "overview"),
          (snap) => {
            if (snap.exists()) {
              state.meta = snap.data();
              renderKpis();
              refreshAllRestFallback().catch(() => {});
            }
          },
          () => {}
        );
        state.unsubs.push(unsubMeta);
        state.liveConnected = true;
        btn.classList.add("connected");
        btn.textContent = "Live Connected (Disconnect)";
        await refreshAllRestFallback();
      } catch (_err) {
        // Fallback: REST runQuery one-shot hydrate if Web SDK blocked
        try {
          const metaDocs = await queryFsCollection("vol2_meta", 1);
          if (metaDocs.length) state.meta = metaDocs[0];
          state.liveConnected = true;
          btn.classList.add("connected");
          btn.textContent = "Live Synced (REST)";
          renderKpis();
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
