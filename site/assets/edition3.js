/**
 * Edition No. 3 Interactive SebeniProofs Stress Observatory & Direct-Deduction Viewer.
 * - Strictly Light Newsprint by default (honoring Evening Edition via CSS vars)
 * - Static-First on GitHub Pages (loads ../../data/edition3*.json with zero Cloud Run wakeups)
 * - Opt-in Connect Live button / ?live=1 wakes Cloud Run /api/ed3/stream SSE
 */

(function () {
  "use strict";

  const esc = (s) => window.Chronicle.esc(s);

  const ALL_14_LANGS = [
    "bam", "bmq", "boz", "dtm", "ful", "hsy", "mku",
    "mlq", "myk", "seq", "ses", "snk", "taq", "bbo",
  ];

  const NIM_FILTER_LIST = [
    ["ALL", "All 4 Models"],
    ["mistral-large-v4", "Mistral Large V4"],
    ["nemotron-3-ultra-550b", "Nemotron-3-Ultra-550B"],
    ["minimax-m3-1-flash", "Minimax M3.1 Flash"],
    ["nemotron-3-5-super-120b", "Nemotron-3.5-Super-120B"],
  ];

  function exactLevDp(ref, hyp) {
    const m = ref.length;
    const n = hyp.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const c = ref[i - 1] === hyp[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c);
      }
    }
    const path = new Set(["0,0"]);
    let i = m;
    let j = n;
    while (i > 0 || j > 0) {
      path.add(`${i},${j}`);
      if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)) {
        i--; j--;
      } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
        i--;
      } else {
        j--;
      }
    }
    path.add("0,0");
    return { dp, path, dist: dp[m][n] };
  }

  function computeIndexMcs(pred, ref) {
    const d = Math.max(pred.length, ref.length);
    if (!d) return 0;
    let hit = 0;
    for (let i = 0; i < Math.min(pred.length, ref.length); i++) {
      if (String(pred[i]) === String(ref[i])) hit++;
    }
    return hit / d;
  }

  function computeAlignedMcs(pred, ref) {
    const d = Math.max(pred.length, ref.length);
    if (!d) return 0;
    const m = pred.length;
    const n = ref.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = String(pred[i - 1]) === String(ref[j - 1])
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    return dp[m][n] / d;
  }

  function computeUwec(stages, pTheta, pRef, beta = 0.1, invert = false) {
    if (!stages.length) return { total: 0, ind: 0, kl: 0 };
    const eps = 1e-8;
    let indSum = 0;
    let klSum = 0;
    for (let i = 0; i < stages.length; i++) {
      const s = Number(stages[i]);
      const indVal = invert ? (s === -1 ? 1 : 0) : (s !== -1 ? 1 : 0);
      const pt = pTheta[i] ?? 0.5;
      const pr = pRef[i] ?? 0.5;
      indSum += indVal;
      klSum += beta * Math.abs(Math.log((pt + eps) / (pr + eps)));
    }
    const ind = indSum / stages.length;
    const kl = klSum / stages.length;
    return { total: ind + kl, ind, kl };
  }

  function checkFsa(tags) {
    let hasStem = false;
    const trace = [];
    let allOk = true;
    for (const raw of tags) {
      const t = String(raw || "").trim().toLowerCase();
      let stepOk = false;
      if (t === "stem") { hasStem = true; stepOk = true; }
      else if (t === "empr") { if (!hasStem) { hasStem = true; stepOk = true; } }
      else if (t === "deriv" || t === "infl") { if (hasStem) stepOk = true; }
      if (!stepOk) allOk = false;
      trace.push({ tag: t || "unk", ok: stepOk });
    }
    return { valid: allOk && hasStem, trace };
  }

  class ProofObservatory {
    constructor(mountEl) {
      this.mount = mountEl;
      this.prefix = mountEl.dataset.prefix || "../../data";
      this.liveOrigin = (mountEl.dataset.liveOrigin || "").replace(/\/$/, "");
      this.apiBase = null;
      this.summary = null;
      this.eventsByLang = {};
      this.runsMeta = [];
      this.healthSnap = null;
      this.selectedLang = "bam";
      this.modelFilter = "ALL";
      this.cursor = 0;
      this.playing = false;
      this.speed = 2;
      this.followLive = false;
      this.sse = null;
      this._statusTimer = null;
      this._renderMount();
      this._init();
    }

    _renderMount() {
      this.mount.innerHTML = `
        <div class="ed3-ribbon" id="ed3Ribbon">
          <span style="font-weight:800">MULTI13 + BBO OOD:</span>
          <span style="color:var(--ink-soft)">Loading stress-test telemetry...</span>
        </div>
        <div class="ed3-toolbar">
          <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
            <label>Language <select id="ed3LangSel"></select></label>
            <label>Model <select id="ed3ModelSel">${NIM_FILTER_LIST.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></label>
            <button type="button" id="ed3Play">Play Stream</button>
            <button type="button" id="ed3Restart">Restart</button>
            <button type="button" id="ed3LiveBtn" style="display:none">Follow Live</button>
            <button type="button" id="ed3ConnectLive" style="display:none">Connect Live</button>
            <button type="button" id="ed3AdminBtn" style="display:none">Admin</button>
          </div>
          <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:200px;max-width:340px">
            <input id="ed3Scrub" type="range" min="0" max="0" value="0" style="flex:1" />
            <span id="ed3StepLbl" style="font-family:var(--mono);font-size:0.72rem;min-width:54px;text-align:right">0 / 0</span>
          </div>
        </div>
        <div class="ed3-admin-drawer" id="ed3AdminDrawer">
          <strong>Admin Trigger (Edition 3):</strong>
          <input id="ed3AdminTok" type="password" placeholder="SWARM_ADMIN_TOKEN" style="width:170px" />
          <button type="button" id="ed3TrigLang">Re-run Lang</button>
          <button type="button" id="ed3TrigAll">Run Pending 14</button>
          <button type="button" id="ed3TrigStop">Stop</button>
          <span id="ed3AdminMsg" style="font-family:var(--mono);font-size:0.72rem"></span>
        </div>

        <div class="ed3-kpi-grid" id="ed3Kpis"></div>

        <div class="ed3-split">
          <div class="ed3-panel">
            <h4>
              <span>Live &amp; Replayable Direct Deductions Stream</span>
              <span id="ed3LangTag" class="tag api">BAM</span>
            </h4>
            <div class="ed3-feed" id="ed3Feed">
              <div class="ed3-ev"><div class="ed3-ev-head"><span>Waiting for data...</span></div></div>
            </div>
          </div>

          <div class="ed3-panel">
            <h4>
              <span>Interactive SebeniProofs Stress Workbench</span>
              <span class="tag sebeni">Lean 4 Counterparts</span>
            </h4>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;font-family:var(--sans);font-size:0.74rem">
              <label>Sample <select id="ed3SpotSel"></select></label>
              <label>Stress Regime
                <select id="ed3RegimeSel">
                  <option value="leading_shift">1. Leading Shift (mcs_shift_collapse vs stageLcs)</option>
                  <option value="overgeneration_drift">2. Over-Generation Drift (MER &gt; 1 vs bounded_mer)</option>
                  <option value="morphotactic_corruption">3. Morphotactic Break (Phi inflation vs FSA)</option>
                  <option value="ood_cross_family">4. Cross-Family / bbo OOD (UWEC orig vs inv)</option>
                  <option value="truncation_asymmetry">5. Truncation Bias (mcs_corpus vs sentence mcs)</option>
                  <option value="clean_identity">6. Clean Identity Baseline</option>
                </select>
              </label>
            </div>
            <div id="ed3WorkbenchBody"></div>
          </div>
        </div>

        <div class="ed3-panel">
          <h4>
            <span>4-Model Direct Deductions &amp; Oracle-Verified Linguistic Proof Cards</span>
            <span id="ed3VerdictBadge" class="tag good">Loading...</span>
          </h4>
          <div class="ed3-deduction-grid" id="ed3ModelCards"></div>
        </div>
      `;

      this.$ribbon = this.mount.querySelector("#ed3Ribbon");
      this.$langSel = this.mount.querySelector("#ed3LangSel");
      this.$modelSel = this.mount.querySelector("#ed3ModelSel");
      this.$play = this.mount.querySelector("#ed3Play");
      this.$restart = this.mount.querySelector("#ed3Restart");
      this.$liveBtn = this.mount.querySelector("#ed3LiveBtn");
      this.$connectLive = this.mount.querySelector("#ed3ConnectLive");
      this.$adminBtn = this.mount.querySelector("#ed3AdminBtn");
      this.$adminDrawer = this.mount.querySelector("#ed3AdminDrawer");
      this.$adminTok = this.mount.querySelector("#ed3AdminTok");
      this.$adminMsg = this.mount.querySelector("#ed3AdminMsg");
      this.$scrub = this.mount.querySelector("#ed3Scrub");
      this.$stepLbl = this.mount.querySelector("#ed3StepLbl");
      this.$kpis = this.mount.querySelector("#ed3Kpis");
      this.$feed = this.mount.querySelector("#ed3Feed");
      this.$langTag = this.mount.querySelector("#ed3LangTag");
      this.$spotSel = this.mount.querySelector("#ed3SpotSel");
      this.$regimeSel = this.mount.querySelector("#ed3RegimeSel");
      this.$workbench = this.mount.querySelector("#ed3WorkbenchBody");
      this.$modelCards = this.mount.querySelector("#ed3ModelCards");
      this.$verdictBadge = this.mount.querySelector("#ed3VerdictBadge");

      const savedTok = sessionStorage.getItem("swarm_admin_tok") || "";
      if (savedTok) this.$adminTok.value = savedTok;
      this.$adminBtn.addEventListener("click", () => this.$adminDrawer.classList.toggle("open"));
      this.$adminTok.addEventListener("input", () => sessionStorage.setItem("swarm_admin_tok", this.$adminTok.value));

      this.mount.querySelector("#ed3TrigLang").addEventListener("click", () =>
        this._adminPost("/api/ed3/admin/trigger", { langs: [this.selectedLang], resume: false })
      );
      this.mount.querySelector("#ed3TrigAll").addEventListener("click", () =>
        this._adminPost("/api/ed3/admin/trigger", { langs: "all", resume: true })
      );
      this.mount.querySelector("#ed3TrigStop").addEventListener("click", () =>
        this._adminPost("/api/ed3/admin/stop", {})
      );

      this.$langSel.addEventListener("change", () => this._switchLang(this.$langSel.value));
      this.$modelSel.addEventListener("change", () => {
        this.modelFilter = this.$modelSel.value;
        this._renderFeed();
      });
      this.$spotSel.addEventListener("change", () => this._renderWorkbench());
      this.$regimeSel.addEventListener("change", () => this._renderWorkbench());

      this.$play.addEventListener("click", () => {
        this.playing = !this.playing;
        this.$play.textContent = this.playing ? "Pause" : "Play Stream";
        if (this.playing) this._tickPlayback();
      });
      this.$restart.addEventListener("click", () => {
        this.followLive = false;
        this.cursor = Math.min(1, (this.eventsByLang[this.selectedLang] || []).length);
        this.playing = true;
        this.$play.textContent = "Pause";
        this._renderFeed();
        this._tickPlayback();
      });
      this.$scrub.addEventListener("input", () => {
        this.playing = false;
        this.$play.textContent = "Play Stream";
        this.cursor = Number(this.$scrub.value) || 0;
        this._renderFeed();
      });
      this.$liveBtn.addEventListener("click", () => {
        this.followLive = !this.followLive;
        this.$liveBtn.classList.toggle("active", this.followLive);
        if (this.followLive) this._connectSse(this.selectedLang);
        else if (this.sse) { this.sse.close(); this.sse = null; }
      });
      this.$connectLive.addEventListener("click", () => this._onConnectLive());
    }

    async _fetchJsonTimeout(url, ms = 3500) {
      const ctrl = new AbortController();
      const tid = setTimeout(() => ctrl.abort(), ms);
      try {
        const r = await fetch(url, { cache: "no-store", signal: ctrl.signal });
        return r.ok ? await r.json() : null;
      } catch (_) {
        return null;
      } finally {
        clearTimeout(tid);
      }
    }

    async _tryLiveBase(base, timeoutMs = 3500) {
      const h = await this._fetchJsonTimeout(`${base}/api/ed3/health`, timeoutMs);
      if (!h || h.status !== "ok") return false;
      const rr = await this._fetchJsonTimeout(`${base}/api/ed3/runs`, timeoutMs);
      if (!rr || !Array.isArray(rr.runs)) return false;
      const sum = await this._fetchJsonTimeout(`${base}/api/ed3/edition3.json`, timeoutMs);
      if (sum && sum.languages) this.summary = sum;

      this.apiBase = base;
      this.healthSnap = h;
      this.runsMeta = rr.runs;
      this.$liveBtn.style.display = "inline-block";
      this.$adminBtn.style.display = "inline-block";
      this.$connectLive.style.display = "none";

      const active = rr.runs.find((r) => r.active);
      const firstDone = rr.runs.find((r) => r.done || Number(r.events) > 0);
      this.selectedLang = String((active || firstDone || rr.runs[0]).lang);
      this.followLive = true;
      this.$liveBtn.classList.add("active");
      this._populateSelectors();
      this._renderSummaryTables();
      this._connectSse(this.selectedLang);
      if (this._statusTimer) clearInterval(this._statusTimer);
      this._statusTimer = setInterval(() => this._refreshLiveStatus(), 15000);
      return true;
    }

    async _refreshLiveStatus() {
      if (this.apiBase === null) return;
      const [h, rr, sum] = await Promise.all([
        this._fetchJsonTimeout(`${this.apiBase}/api/ed3/health`, 3000),
        this._fetchJsonTimeout(`${this.apiBase}/api/ed3/runs`, 3000),
        this._fetchJsonTimeout(`${this.apiBase}/api/ed3/edition3.json`, 3000),
      ]);
      if (h) this.healthSnap = h;
      if (rr && Array.isArray(rr.runs)) this.runsMeta = rr.runs;
      if (sum && sum.languages) {
        this.summary = sum;
        this._renderSummaryTables();
        this._renderLanguageView();
      }
      this._renderRibbon();
    }

    async _onConnectLive() {
      if (!this.liveOrigin) return;
      this.$connectLive.disabled = true;
      this.$connectLive.textContent = "Waking Cloud Run...";
      const ok = await this._tryLiveBase(this.liveOrigin, 15000);
      if (!ok) {
        this.$connectLive.disabled = false;
        this.$connectLive.textContent = "Retry Connect Live";
      }
    }

    async _adminPost(path, body) {
      if (this.apiBase === null) return;
      const tok = (this.$adminTok.value || "").trim();
      if (!tok) { this.$adminMsg.textContent = "Enter admin token first."; return; }
      try {
        const r = await fetch(`${this.apiBase}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${tok}` },
          body: JSON.stringify(body),
        });
        const j = await r.json();
        this.$adminMsg.textContent = r.ok ? JSON.stringify(j) : `HTTP ${r.status}: ${j.error || "failed"}`;
        this._refreshLiveStatus();
      } catch (e) {
        this.$adminMsg.textContent = e.message;
      }
    }

    async _init() {
      const wantLive = new URLSearchParams(window.location.search).get("live") === "1";
      const isSameLive = Boolean(this.liveOrigin && window.location.origin.startsWith(this.liveOrigin));
      const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

      const candidates = [];
      if (isSameLive || isLocal) candidates.push({ base: "", timeout: 3000 });
      if (wantLive && this.liveOrigin && !isSameLive) candidates.push({ base: this.liveOrigin, timeout: 15000 });

      for (const c of candidates) {
        if (await this._tryLiveBase(c.base, c.timeout)) return;
      }

      if (this.liveOrigin && !isSameLive) {
        this.$connectLive.style.display = "inline-block";
      }

      const [sum, evDoc] = await Promise.all([
        this._fetchJsonTimeout(`${this.prefix}/edition3.json`, 6000),
        this._fetchJsonTimeout(`${this.prefix}/edition3_events.json`, 6000),
      ]);
      if (!sum && this.liveOrigin && !isSameLive) {
        if (await this._tryLiveBase(this.liveOrigin, 15000)) return;
      }
      if (sum) this.summary = sum;
      if (evDoc && evDoc.events_by_lang) this.eventsByLang = evDoc.events_by_lang;

      const langs = (sum && sum.langs && sum.langs.length)
        ? sum.langs
        : (evDoc && evDoc.langs) || ALL_14_LANGS;
      this.selectedLang = langs[0] || "bam";
      this._populateSelectors();
      this._renderSummaryTables();
      this._switchLang(this.selectedLang);
    }

    _populateSelectors() {
      const langs = this.summary?.langs?.length
        ? this.summary.langs
        : (this.runsMeta.length ? this.runsMeta.map((r) => r.lang) : ALL_14_LANGS);
      this.$langSel.innerHTML = langs
        .map((l) => {
          const ldoc = this.summary?.languages?.[l]?.corpus;
          const label = ldoc ? `${l.toUpperCase()} — ${ldoc.lang_name}${ldoc.is_ood ? " [OOD]" : ""}` : l.toUpperCase();
          return `<option value="${esc(l)}" ${l === this.selectedLang ? "selected" : ""}>${esc(label)}</option>`;
        })
        .join("");
      this._renderRibbon();
    }

    _renderRibbon() {
      const langs = this.runsMeta.length ? this.runsMeta : ALL_14_LANGS.map((l) => ({
        lang: l,
        done: Boolean(this.summary?.languages?.[l]),
        active: false,
      }));
      const doneCnt = langs.filter((r) => r.done).length;
      const mode = this.apiBase !== null
        ? (this.healthSnap?.running ? `LIVE · running ${(this.healthSnap.active_lang || "").toUpperCase()}` : "LIVE SERVER READY")
        : "STATIC REPLAY";
      this.$ribbon.innerHTML = [
        `<span style="font-weight:800;margin-right:4px">MULTI13 + BBO (${doneCnt}/${langs.length} · ${esc(mode)}):</span>`,
        ...langs.map((r) => {
          const cls = [
            "ed3-lang-pill",
            r.done ? "done" : "",
            r.active ? "live" : "",
            r.lang === "bbo" ? "ood" : "",
            r.lang === this.selectedLang ? "selected" : "",
          ].filter(Boolean).join(" ");
          const mark = r.active ? "●" : r.done ? "✓" : "○";
          const oodTag = r.lang === "bbo" ? " OOD" : "";
          return `<button type="button" class="${cls}" data-pick="${esc(r.lang)}">${esc(r.lang.toUpperCase())}${oodTag} ${mark}</button>`;
        }),
      ].join("");

      this.$ribbon.querySelectorAll("[data-pick]").forEach((btn) => {
        btn.addEventListener("click", () => {
          this.$langSel.value = btn.dataset.pick;
          this._switchLang(btn.dataset.pick);
        });
      });
    }

    _connectSse(lang) {
      if (this.sse) { this.sse.close(); this.sse = null; }
      this.eventsByLang[lang] = [];
      this.cursor = 0;
      const sse = new EventSource(`${this.apiBase}/api/ed3/stream?lang=${encodeURIComponent(lang)}`);
      sse.onmessage = (msg) => {
        try {
          const ev = JSON.parse(msg.data);
          (this.eventsByLang[lang] = this.eventsByLang[lang] || []).push(ev);
          this.cursor = this.eventsByLang[lang].length;
          this._renderFeed();
        } catch (_) {}
      };
      this.sse = sse;
    }

    _switchLang(lang) {
      this.selectedLang = lang;
      this.$langTag.textContent = lang === "bbo" ? "BBO (OOD CONTROL)" : lang.toUpperCase();
      this._renderRibbon();
      if (this.followLive && this.apiBase !== null) {
        this._connectSse(lang);
      } else {
        const evs = this.eventsByLang[lang] || [];
        this.cursor = evs.length;
        this._renderFeed();
      }
      this._renderLanguageView();
    }

    _tickPlayback() {
      if (!this.playing || this.followLive) return;
      const evs = this.eventsByLang[this.selectedLang] || [];
      if (this.cursor >= evs.length) {
        this.playing = false;
        this.$play.textContent = "Play Stream";
        return;
      }
      this.cursor += 1;
      this._renderFeed();
      setTimeout(() => this._tickPlayback(), 650);
    }

    _renderFeed() {
      const evs = this.eventsByLang[this.selectedLang] || [];
      this.$scrub.max = String(evs.length);
      this.$scrub.value = String(this.cursor);
      this.$stepLbl.textContent = `${this.cursor} / ${evs.length}`;

      const shown = evs.slice(0, this.cursor).filter(
        (e) => this.modelFilter === "ALL" || e.model_slug === this.modelFilter || e.model_slug === "dabax-oracle" || e.model_slug === "sebeni-proofs"
      );
      if (!shown.length) {
        this.$feed.innerHTML = `<div class="ed3-ev"><div class="ed3-ev-head"><span>Awaiting events for ${esc(this.selectedLang.toUpperCase())}...</span></div></div>`;
        return;
      }

      this.$feed.innerHTML = shown.map((ev) => {
        const traceHtml = ev.reasoning_excerpt
          ? `<details class="ed3-trace" ${ev.kind === "model_deduction" ? "open" : ""}>
               <summary>Direct Deduction Trace (${ev.reasoning_excerpt.length} chars)</summary>
               <pre>${esc(ev.reasoning_excerpt)}</pre>
             </details>`
          : "";
        return `
          <div class="ed3-ev kind-${esc(ev.kind)}">
            <div class="ed3-ev-head">
              <strong>#${ev.seq} &bull; ${esc(ev.model_name)} &bull; ${esc(ev.kind)}</strong>
              <span>${esc(ev.label)}</span>
            </div>
            <div>${esc(ev.summary)}</div>
            ${traceHtml}
          </div>
        `;
      }).join("");
      this.$feed.scrollTop = this.$feed.scrollHeight;
    }

    _renderLanguageView() {
      const lres = this.summary?.languages?.[this.selectedLang];
      const metaEl = document.getElementById("ed3MetaBanner");
      if (metaEl && this.summary) {
        const g = this.summary.global_metrics || {};
        metaEl.textContent = `EDITION NO. 3 • ${this.summary.langs.length} LANGUAGES (${this.summary.total_sentences} SENTENCES) • COHEN d(MER)=${g.mean_cohens_d_mer_corrupt ?? "-"} • OOD GAP=${g.mean_ood_strict_gap_pct ?? "-"} pp`;
      }
      if (!lres) {
        this.$kpis.innerHTML = `<div class="ed3-kpi"><div class="k">Status</div><div class="v">Awaiting ${esc(this.selectedLang.toUpperCase())}</div><div class="s">Sync or click Connect Live.</div></div>`;
        this.$workbench.innerHTML = "<p>Awaiting corpus stress data.</p>";
        this.$modelCards.innerHTML = "<p>Awaiting 4-model deduction results.</p>";
        return;
      }

      const c = lres.corpus;
      this.$kpis.innerHTML = `
        <div class="ed3-kpi ${c.is_ood ? "warn" : "good"}">
          <div class="k">Baseline Strict Parse</div>
          <div class="v">${c.clean_strict_parse_pct.toFixed(1)}%</div>
          <div class="s">${c.n_samples} rows &bull; DabaX(<code>${esc(c.dabax_baseline)}</code>)${c.is_ood ? " (OOD)" : ""}</div>
        </div>
        <div class="ed3-kpi good">
          <div class="k">MER Separation (Cohen d)</div>
          <div class="v">${c.cohens_d_mer_corrupt.toFixed(2)}</div>
          <div class="s">Clean vs. Morphotactic Corruption</div>
        </div>
        <div class="ed3-kpi">
          <div class="k">MCS Shift Recovery</div>
          <div class="v">+${c.shift_mcs_recovery_pct.toFixed(1)} pp</div>
          <div class="s"><code>stageLcs</code> over index-MCS (d=${c.cohens_d_mcs_shift.toFixed(2)})</div>
        </div>
        <div class="ed3-kpi bad">
          <div class="k">MER &gt; 1 Saturation Prevented</div>
          <div class="v">${c.mer_saturation_prevented_pct.toFixed(1)}%</div>
          <div class="s">Caught by <code>bounded_mer &le; 1.0</code></div>
        </div>
        <div class="ed3-kpi good">
          <div class="k">UWEC OOD Separation</div>
          <div class="v">d=${c.cohens_d_uwec_ood.toFixed(2)}</div>
          <div class="s">Cross-family strict gap: ${c.ood_strict_gap_pct >= 0 ? "+" : ""}${c.ood_strict_gap_pct.toFixed(1)} pp</div>
        </div>
        <div class="ed3-kpi">
          <div class="k">Oracle &amp; Model Consensus</div>
          <div class="v">${lres.consensus_effectiveness.toFixed(1)}/100</div>
          <div class="s">${lres.oracle_pass_rate_pct.toFixed(0)}% live DabaX probes verified</div>
        </div>
      `;

      const spots = c.spotlights || [];
      this.$spotSel.innerHTML = spots
        .map((sp, i) => `<option value="${i}">Sample #${sp.idx + 1} (${sp.ref_morphemes.length} morphemes)</option>`)
        .join("");
      this._renderWorkbench();

      this.$verdictBadge.textContent = `${lres.consensus_verdict} · ${lres.consensus_effectiveness.toFixed(1)}/100`;
      this.$modelCards.innerHTML = (lres.models || []).map((m) => `
        <div class="ed3-model-card verdict-${esc(m.verdict)}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <strong>${esc(m.model_name)}</strong>
            <span class="tag ${m.verdict === "PROVEN_EFFECTIVE" ? "sebeni" : "slm"}">${esc(m.verdict)} (${m.effectiveness_score.toFixed(0)})</span>
          </div>
          <div style="font-family:var(--mono);font-size:0.7rem;color:var(--ink-muted);margin-bottom:6px">
            ${esc(m.params_label)} &bull; R1 ${m.round1_latency_s}s / R2 ${m.round2_latency_s}s &bull; Oracle ${m.oracle_verified_count}/${m.oracle_total_count}
          </div>
          <p style="margin:0 0 5px"><strong>MER:</strong> ${esc(m.mer_assessment || m.round1_summary)}</p>
          <p style="margin:0 0 5px"><strong>MCS:</strong> ${esc(m.mcs_assessment)}</p>
          <p style="margin:0 0 6px"><strong>UWEC:</strong> ${esc(m.uwec_assessment)}</p>
          ${m.probes?.length ? `<div style="font-family:var(--mono);font-size:0.7rem;background:var(--paper);border:1px solid var(--rule-hair);padding:5px;margin-bottom:6px">
            ${m.probes.map((p) => `<div><strong>[${p.direction_verified ? "ORACLE PASS" : "CHECK"}]</strong> <code>${esc(p.probe_name)}</code>: MER=${p.oracle_mer_bounded}, MCS=${p.oracle_mcs_index}&rarr;${p.oracle_mcs_aligned}, UWEC_inv=${p.oracle_uwec_inv}</div>`).join("")}
          </div>` : ""}
          ${m.round2_reasoning ? `<details class="ed3-trace"><summary>Full Direct Deduction (${m.round2_reasoning.length} chars)</summary><pre>${esc(m.round2_reasoning)}</pre></details>` : ""}
        </div>
      `).join("");
    }

    _renderWorkbench() {
      const lres = this.summary?.languages?.[this.selectedLang];
      const sp = lres?.corpus?.spotlights?.[Number(this.$spotSel.value) || 0];
      if (!sp) return;
      const regime = this.$regimeSel.value || "leading_shift";
      const refM = sp.ref_morphemes || ["a", "b", "c"];
      const refS = sp.ref_stages || [0, 0, 0];

      let hypM = [...refM];
      let predS = [...refS];
      let tags = [...(sp.fsa_tags || ["stem"])];
      let pTheta = refS.map((s) => (s >= 0 ? 0.85 : 0.2));
      let pRef = refS.map(() => 0.5);
      let theoremNote = "";

      if (regime === "leading_shift") {
        hypM = ["<shift>", ...refM];
        predS = [-1, ...refS];
        tags = ["unk", ...tags];
        pTheta = [0.3, ...pTheta];
        pRef = [0.5, ...pRef];
        theoremNote = "Metrics.lean::stageLcs_resists_shift_witness — 1 leading insertion collapses index MCS toward 0 while LCS recovers alignment.";
      } else if (regime === "overgeneration_drift") {
        const extra = Math.max(3, refM.length + 2);
        hypM = [...refM, ...Array.from({ length: extra }, (_, k) => `x${k + 1}`)];
        predS = [...refS, ...Array(extra).fill(-1)];
        tags = [...tags, ...Array(extra).fill("unk")];
        pTheta = Array(hypM.length).fill(0.9);
        pRef = Array(hypM.length).fill(0.2);
        theoremNote = "Metrics.lean::bounded_mer_always_unit_bounded — |hyp| > 2|ref| forces raw MER > 1.0, whereas bounded_mer stays in [0, 1].";
      } else if (regime === "morphotactic_corruption") {
        hypM = refM.length >= 2 ? [...refM.slice(1).reverse(), "-sfx?"] : ["-sfx?", "unk"];
        predS = hypM.map((_, k) => (k % 2 === 0 ? -1 : 0));
        tags = ["infl", ...refM.slice(1).map(() => "stem")];
        pTheta = hypM.map(() => 0.75);
        pRef = hypM.map(() => 0.45);
        theoremNote = "Morphotactics.lean::phi_inflation_without_fsa_validity — Bare inflection without preceding stem is rejected by MorphotacticFSA.";
      } else if (regime === "ood_cross_family") {
        hypM = (sp.ood_cross_morphemes && sp.ood_cross_morphemes.length) ? sp.ood_cross_morphemes : refM;
        predS = (sp.ood_cross_stages && sp.ood_cross_stages.length) ? sp.ood_cross_stages : refS.map(() => -1);
        tags = predS.map((s) => (s < 0 ? "unk" : "empr"));
        pTheta = predS.map((s) => (s < 0 ? 0.25 : 0.75));
        pRef = predS.map(() => 0.75);
        theoremNote = `Metrics.lean::uwec_current_indicator_prefers_unknown — Cross-family (${lres.corpus.ood_cross_baseline}) parser exposes unknown stages under inverted UWEC.`;
      } else if (regime === "truncation_asymmetry") {
        const keep = Math.max(1, Math.floor(refM.length / 2));
        hypM = refM.slice(0, keep);
        predS = refS.slice(0, keep);
        pTheta = predS.map(() => 0.8);
        pRef = predS.map(() => 0.5);
        theoremNote = "Metrics.lean::mcs_corpus_denom_asymmetry — |pred| < |ref| makes mcs_corpus (|pred| denom) strictly higher than sentence mcs.";
      } else {
        theoremNote = "Basic.lean::stageToPhi_nonneg_bound — Identity parse satisfies MER=0, MCS=1, and admissible FSA chain.";
      }

      const dpRes = exactLevDp(refM.slice(0, 8), hypM.slice(0, 8));
      const merRaw = dpRes.dist / Math.max(1, Math.min(8, refM.length));
      const merBnd = dpRes.dist / Math.max(1, Math.min(8, refM.length), Math.min(8, hypM.length));
      const mIdx = computeIndexMcs(predS, refS);
      const mAln = computeAlignedMcs(predS, refS);
      const uOrig = computeUwec(predS, pTheta, pRef, 0.1, false);
      const uInv = computeUwec(predS, pTheta, pRef, 0.1, true);
      const fsa = checkFsa(tags.slice(0, 10));
      const rSub = refM.slice(0, 8);
      const hSub = hypM.slice(0, 8);

      const headerCells = ["<th>DP</th>", "<th>&epsilon;</th>", ...hSub.map((w) => `<th>${esc(w)}</th>`)].join("");
      const dpRows = dpRes.dp.map((row, i) => {
        const lbl = i === 0 ? "&epsilon;" : esc(rSub[i - 1]);
        const tds = row.map((v, j) => `<td class="${dpRes.path.has(`${i},${j}`) ? "dp-path" : ""}">${v}</td>`).join("");
        return `<tr><th>${lbl}</th>${tds}</tr>`;
      }).join("");

      const uTot = Math.max(1e-6, uInv.total);
      const indPct = (100 * uInv.ind) / uTot;
      const klPct = (100 * uInv.kl) / uTot;

      this.$workbench.innerHTML = `
        <div style="font-size:0.8rem;background:var(--paper);border:1px solid var(--rule-hair);padding:8px 10px;margin-bottom:10px">
          <strong>Text (#${sp.idx + 1}):</strong> <em>${esc(sp.text)}</em><br/>
          <span style="color:var(--crimson);font-family:var(--mono);font-size:0.72rem">${esc(theoremNote)}</span>
        </div>
        <div class="ed3-kpi-grid" style="grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:10px">
          <div class="ed3-kpi"><div class="k">MER Raw &rarr; Bnd</div><div class="v">${merRaw.toFixed(2)} &rarr; ${merBnd.toFixed(2)}</div></div>
          <div class="ed3-kpi"><div class="k">MCS Idx &rarr; LCS</div><div class="v">${mIdx.toFixed(2)} &rarr; ${mAln.toFixed(2)}</div></div>
          <div class="ed3-kpi"><div class="k">UWEC Orig / Inv</div><div class="v">${uOrig.total.toFixed(2)} / ${uInv.total.toFixed(2)}</div></div>
          <div class="ed3-kpi ${fsa.valid ? "good" : "bad"}"><div class="k">FSA Status</div><div class="v">${fsa.valid ? "VALID" : "REJECTED"}</div></div>
        </div>
        <div style="font-family:var(--sans);font-size:0.7rem;font-weight:700;text-transform:uppercase;color:var(--ink-soft);margin-bottom:4px">
          Morphotactic FSA Trace
        </div>
        <div class="ed3-fsa-track">
          ${fsa.trace.map((n, idx) => `${idx ? "<span>&rarr;</span>" : ""}<span class="ed3-fsa-node ${n.ok ? "ok" : "bad"}">${esc(n.tag)}</span>`).join("")}
        </div>
        <div style="font-family:var(--sans);font-size:0.7rem;font-weight:700;text-transform:uppercase;color:var(--ink-soft);margin-bottom:2px">
          Inverted UWEC Decomposition: Ind=${uInv.ind.toFixed(3)} + KL=${uInv.kl.toFixed(3)} = ${uInv.total.toFixed(3)}
        </div>
        <div class="ed3-uwec-bar">
          <div class="ed3-uwec-ind" style="width:${indPct.toFixed(1)}%"></div>
          <div class="ed3-uwec-kl" style="width:${klPct.toFixed(1)}%"></div>
        </div>
        <div style="font-family:var(--sans);font-size:0.7rem;font-weight:700;text-transform:uppercase;color:var(--ink-soft);margin:8px 0 4px">
          Levenshtein MER DP Matrix (up to 8&times;8 slice)
        </div>
        <div style="overflow-x:auto">
          <table class="ed3-dp-table"><thead><tr>${headerCells}</tr></thead><tbody>${dpRows}</tbody></table>
        </div>
      `;
    }

    _renderSummaryTables() {
      if (!this.summary) return;
      const g = this.summary.global_metrics || {};

      const ledgerBody = document.getElementById("ed3ModelLedgerBody");
      if (ledgerBody) {
        ledgerBody.innerHTML = (this.summary.model_leaderboard || []).map((r, i) => `
          <tr>
            <td class="num">${i + 1}</td>
            <td><strong>${esc(r.model_name)}</strong><br/><small style="color:var(--ink-soft)"><code>${esc(r.model_id)}</code></small></td>
            <td>${esc(r.params_label)}</td>
            <td class="num">${r.langs_completed} / 14</td>
            <td class="num">${r.oracle_verified} / ${r.oracle_total} (${r.oracle_pass_pct.toFixed(0)}%)</td>
            <td class="num">${r.mean_r1_latency_s.toFixed(1)}s / ${r.mean_r2_latency_s.toFixed(1)}s</td>
            <td><span class="tag ${r.dominant_verdict === "PROVEN_EFFECTIVE" ? "sebeni" : "slm"}">${esc(r.dominant_verdict)}</span></td>
            <td class="num"><strong>${r.mean_effectiveness_score.toFixed(1)}</strong></td>
          </tr>
        `).join("");
      }

      const langBody = document.getElementById("ed3LangMatrixBody");
      if (langBody) {
        langBody.innerHTML = (this.summary.langs || []).map((l) => {
          const lres = this.summary.languages[l];
          const c = lres.corpus;
          const rSh = c.regimes.leading_shift;
          const rOg = c.regimes.overgeneration_drift;
          const rOod = c.regimes.ood_cross_family;
          return `
            <tr data-pick-row="${esc(l)}" style="cursor:pointer">
              <td><strong>${esc(l.toUpperCase())}</strong>${c.is_ood ? ' <span class="tag slm">OOD</span>' : ""}<br/><small>${esc(c.lang_name)}</small></td>
              <td><code>${esc(c.dabax_baseline)}</code></td>
              <td class="num">${c.clean_strict_parse_pct.toFixed(1)}%</td>
              <td class="num"><strong>${c.cohens_d_mer_corrupt.toFixed(2)}</strong></td>
              <td class="num">${rSh ? `${rSh.mean_mcs_index.toFixed(2)} &rarr; ${rSh.mean_mcs_aligned.toFixed(2)}` : "-"}</td>
              <td class="num">${rOg ? `${rOg.mean_mer_raw.toFixed(2)} &rarr; ${rOg.mean_mer_bounded.toFixed(2)}` : "-"}</td>
              <td class="num">${rOod ? `${rOod.mean_uwec_orig.toFixed(2)} / <strong>${rOod.mean_uwec_inv.toFixed(2)}</strong>` : "-"}</td>
              <td class="num">${c.ood_strict_gap_pct >= 0 ? "+" : ""}${c.ood_strict_gap_pct.toFixed(1)} pp</td>
              <td class="num"><strong>${lres.consensus_effectiveness.toFixed(1)}</strong></td>
            </tr>
          `;
        }).join("");
        langBody.querySelectorAll("[data-pick-row]").forEach((tr) => {
          tr.addEventListener("click", () => {
            this.$langSel.value = tr.dataset.pickRow;
            this._switchLang(tr.dataset.pickRow);
            this.mount.scrollIntoView({ behavior: "smooth", block: "start" });
          });
        });
      }

      const thmEl = document.getElementById("ed3LeanTheorems");
      if (thmEl) {
        thmEl.innerHTML = (this.summary.lean_theorems || []).map((t) => `
          <div class="flow-card good" style="margin:0">
            <strong><code>${esc(t.name)}</code></strong> <small style="color:var(--ink-soft)">(${esc(t.file)})</small><br/>
            <code>${esc(t.statement)}</code><br/>
            <span style="font-size:0.8rem">${esc(t.meaning)}</span>
          </div>
        `).join("");
      }

      const chartEl = document.getElementById("ed3DiscrimChart");
      if (chartEl && (this.summary.langs || []).length) {
        const w = 780;
        const h = 250;
        const padL = 46;
        const padB = 34;
        const padT = 18;
        const chartW = w - padL - 20;
        const chartH = h - padT - padB;
        const langs = this.summary.langs;
        const step = chartW / Math.max(1, langs.length);
        const maxD = Math.max(4, ...langs.map((l) => Math.max(
          this.summary.languages[l].corpus.cohens_d_mer_corrupt,
          this.summary.languages[l].corpus.cohens_d_uwec_ood
        )));
        const bars = langs.map((l, idx) => {
          const c = this.summary.languages[l].corpus;
          const x = padL + idx * step + 6;
          const bw = Math.max(6, (step - 16) / 2);
          const h1 = (c.cohens_d_mer_corrupt / maxD) * chartH;
          const h2 = (c.cohens_d_uwec_ood / maxD) * chartH;
          return `
            <rect x="${x.toFixed(1)}" y="${(padT + chartH - h1).toFixed(1)}" width="${bw.toFixed(1)}" height="${h1.toFixed(1)}" fill="#8c1d24" />
            <rect x="${(x + bw + 2).toFixed(1)}" y="${(padT + chartH - h2).toFixed(1)}" width="${bw.toFixed(1)}" height="${h2.toFixed(1)}" fill="#1b4d6e" />
            <text x="${(x + bw).toFixed(1)}" y="${h - 12}" text-anchor="middle" font-size="9" font-family="monospace" fill="#181613">${esc(l.toUpperCase())}</text>
          `;
        }).join("");
        chartEl.innerHTML = `
          <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + chartH}" stroke="#1c1915" />
          <line x1="${padL}" y1="${padT + chartH}" x2="${w - 16}" y2="${padT + chartH}" stroke="#1c1915" />
          <text x="${padL - 6}" y="${padT + 8}" text-anchor="end" font-size="9" font-family="monospace" fill="#554f47">${maxD.toFixed(1)}σ</text>
          <text x="${padL - 6}" y="${padT + chartH}" text-anchor="end" font-size="9" font-family="monospace" fill="#554f47">0</text>
          ${bars}
        `;
      }

      const museNoteEl = document.getElementById("ed3MuseNote");
      if (museNoteEl && this.summary.muse_eol_note) {
        museNoteEl.textContent = this.summary.muse_eol_note;
      }

      const statBox = document.getElementById("ed3TakeawaysBox");
      if (statBox && g.n_langs) {
        statBox.innerHTML = `
          <h3>Key Empirical Proof Takeaways (${g.n_langs} Codes &bull; ${g.total_sentences} Sentences &bull; ${g.total_tokens} Tokens)</h3>
          <div class="cols-3" style="margin-top:10px">
            <div class="flow-card good">
              <strong>1. MER Separates Linguistic Corruption (d=${g.mean_cohens_d_mer_corrupt})</strong><br/>
              Across ${g.total_sentences} real <code>dataset_300_samples.jsonl</code> sentences, <code>MER</code> cleanly separates valid DabaX morphology from stem/affix breakages, while <code>bounded_mer &le; 1.0</code> prevented saturation on <strong>${g.mean_mer_saturation_prevented_pct}%</strong> of over-generated hypotheses (<code>bounded_mer_always_unit_bounded</code>).
            </div>
            <div class="flow-card">
              <strong>2. Aligned MCS Resists Shifts (+${g.mean_shift_mcs_recovery_pct} pp)</strong><br/>
              A single leading clitic or OOV insertion collapsed index-paired <code>MCS</code> (<code>mcs_shift_collapse_witness</code>), while LCS-aligned <code>stageLcs</code> recovered <strong>+${g.mean_shift_mcs_recovery_pct} pp</strong> stage consistency and exposed <strong>+${g.mean_truncation_denom_bias_pct} pp</strong> truncation bias under corpus denominators.
            </div>
            <div class="flow-card note">
              <strong>3. Hardened UWEC Isolates OOD Drift (d=${g.mean_cohens_d_uwec_ood})</strong><br/>
              Cross-family parses and <code>bbo</code> (<code>${g.ood_bbo_strict_parse_pct}%</code> zero-shot strict parse vs. <strong>${g.mean_clean_strict_parse_pct}%</strong> in-domain MALI13, a <strong>${g.mean_ood_strict_gap_pct} pp</strong> gap) prove inverted <code>UWEC</code> + <code>allowPromoteStrictOOD</code> are indispensable guards for MALI13 evaluation.
            </div>
          </div>
        `;
      }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const el = document.getElementById("ed3ObservatoryMount");
    if (el) new ProofObservatory(el);
  });
})();
