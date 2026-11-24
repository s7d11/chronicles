(function () {
  const { esc, fetchJson } = window.Chronicle;

  let PAYLOAD = null;
  let SAMPLES = null;
  let lbFilter = "ALL";
  let lbSort = "q";
  let lbAsc = false;
  let paretoAxis = "fert";
  let heatMetric = "q";
  let lidModel = null;
  let inspLang = "taq";

  function renderLeaderboard() {
    const models = PAYLOAD.models || [];
    let rows = models.filter((m) => lbFilter === "ALL" || m.track === lbFilter);
    rows.sort((a, b) => {
      const va = a[lbSort] ?? (lbAsc ? 99999 : -99999);
      const vb = b[lbSort] ?? (lbAsc ? 99999 : -99999);
      const cmp = typeof va === "string" ? va.localeCompare(vb) : va - vb;
      return lbAsc ? cmp : -cmp;
    });

    document.getElementById("lbBody").innerHTML = rows.map((m, idx) => `
      <tr>
        <td class="num">${idx + 1}</td>
        <td><strong>${esc(m.id)}</strong></td>
        <td><span class="tag ${esc(m.trackCls)}">${esc(m.track)}</span></td>
        <td class="num"><strong>${m.q.toFixed(1)}</strong></td>
        <td class="num">${m.q_min3.toFixed(1)}</td>
        <td class="num">${m.morph.toFixed(1)}</td>
        <td class="num">${m.gen.toFixed(1)}</td>
        <td class="num">${m.lid.toFixed(1)}</td>
        <td class="num">${m.cloze.toFixed(1)}</td>
        <td class="num">${m.fert.toFixed(2)}</td>
        <td class="num">${m.diac.toFixed(2)}&times;</td>
        <td class="num">${m.bpb === null || m.bpb === undefined ? "n/a" : m.bpb.toFixed(3)}</td>
        <td class="num">${m.lat.toFixed(2)}s</td>
        <td class="num">${esc(m.compute)}</td>
      </tr>
    `).join("");
  }

  function renderPareto() {
    const models = PAYLOAD.models || [];
    const cfgMap = {
      fert: {
        label: "Word Fertility (subwords/word, lower = better)",
        fn: (m) => m.fert,
        fmt: (v) => v.toFixed(2),
      },
      lat: {
        label: "Mean Latency (s/sample, lower = faster)",
        fn: (m) => m.lat,
        fmt: (v) => v.toFixed(2) + "s",
      },
      params: {
        label: "Active Parameters log10(B + 0.1) (models with known param counts)",
        fn: (m) => (m.params === null || m.params === undefined ? null : Math.log10(m.params + 0.1)),
        fmt: (v) => v.toFixed(2),
      },
      cost: {
        label: "Estimated Run Cost ($ USD, local / free tier = $0)",
        fn: (m) => m.cost,
        fmt: (v) => "$" + v.toFixed(2),
      },
    };
    const cfg = cfgMap[paretoAxis];
    document.getElementById("paretoXTitle").textContent = cfg.label;

    const pts = models
      .map((m) => ({ m, xVal: cfg.fn(m), yVal: m.q }))
      .filter((p) => p.xVal !== null && Number.isFinite(p.xVal));

    if (!pts.length) {
      document.getElementById("paretoSvg").innerHTML =
        `<text x="340" y="170" text-anchor="middle" fill="var(--ink-soft)" font-family="sans-serif">No points available for this axis.</text>`;
      document.getElementById("paretoFrontierList").textContent = "No points on this axis.";
      return;
    }

    const xs = pts.map((p) => p.xVal);
    const ys = pts.map((p) => p.yVal);
    const xMinRaw = Math.min(...xs);
    const xMaxRaw = Math.max(...xs);
    const xPad = Math.max(0.08, (xMaxRaw - xMinRaw) * 0.15);
    const xMin = xMinRaw - xPad;
    const xMax = xMaxRaw + xPad;

    const yMin = Math.max(0, Math.floor((Math.min(...ys) - 6) / 5) * 5);
    const yMax = Math.min(100, Math.ceil((Math.max(...ys) + 6) / 5) * 5);

    const sorted = [...pts].sort((a, b) => a.xVal - b.xVal || b.yVal - a.yVal);
    const frontier = [];
    let bestY = -1;
    for (const p of sorted) {
      if (p.yVal > bestY) {
        frontier.push(p);
        bestY = p.yVal;
      }
    }
    const fIds = new Set(frontier.map((p) => p.m.slug));
    document.getElementById("paretoFrontierList").textContent =
      "Non-dominated Pareto models:\n• " +
      frontier.map((p) => `${p.m.id} (Q=${p.yVal.toFixed(1)}, x=${cfg.fmt(p.xVal)})`).join("\n• ");

    const W = 720, H = 370, padL = 56, padR = 36, padT = 26, padB = 46;
    const mapX = (x) => padL + ((x - xMin) / Math.max(1e-6, xMax - xMin)) * (W - padL - padR);
    const mapY = (y) => H - padB - ((y - yMin) / Math.max(1e-6, yMax - yMin)) * (H - padT - padB);

    let svg = "";
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (i / 4) * (yMax - yMin);
      const py = mapY(yv);
      svg += `<line x1="${padL}" y1="${py}" x2="${W - padR}" y2="${py}" stroke="var(--rule-hair)" stroke-dasharray="3,3" />`;
      svg += `<text x="${padL - 8}" y="${py + 4}" text-anchor="end" font-size="11" fill="var(--ink-soft)" font-family="monospace">${yv.toFixed(0)}</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const xv = xMin + (i / 4) * (xMax - xMin);
      const px = mapX(xv);
      svg += `<line x1="${px}" y1="${padT}" x2="${px}" y2="${H - padB}" stroke="var(--rule-hair)" stroke-dasharray="3,3" />`;
      svg += `<text x="${px}" y="${H - padB + 17}" text-anchor="middle" font-size="11" fill="var(--ink-soft)" font-family="monospace">${cfg.fmt(xv)}</text>`;
    }
    svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="var(--rule-heavy)" stroke-width="1.5"/>`;
    svg += `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="var(--rule-heavy)" stroke-width="1.5"/>`;
    svg += `<text x="16" y="${H / 2}" transform="rotate(-90 16 ${H / 2})" text-anchor="middle" font-size="11" fill="var(--ink)" font-family="sans-serif" font-weight="700">MALI13-Q</text>`;

    if (frontier.length >= 2) {
      const d = frontier.map((p, idx) => `${idx === 0 ? "M" : "L"}${mapX(p.xVal).toFixed(1)},${mapY(p.yVal).toFixed(1)}`).join(" ");
      svg += `<path d="${d}" fill="none" stroke="var(--crimson)" stroke-width="2" stroke-dasharray="6,4" />`;
    }

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const cx = mapX(p.xVal);
      const cy = mapY(p.yVal);
      const col = p.m.track === "Sebeni" ? "var(--forest)" : (p.m.track === "Google" ? "var(--slate)" : "var(--ochre)");
      const ring = fIds.has(p.m.slug) ? `<circle cx="${cx}" cy="${cy}" r="10" fill="none" stroke="var(--crimson)" stroke-width="1.5"/>` : "";
      const align = cx > W - 190 ? "end" : "start";
      const dx = align === "end" ? -10 : 10;
      const dy = (i % 2 === 0 ? -5 : 12);
      svg += `${ring}<circle cx="${cx}" cy="${cy}" r="5.5" fill="${col}" stroke="var(--paper)" stroke-width="1.5"><title>${esc(p.m.id)}: Q=${p.yVal.toFixed(1)}, x=${cfg.fmt(p.xVal)}</title></circle>`;
      svg += `<text x="${cx + dx}" y="${cy + dy}" text-anchor="${align}" font-size="10.5" fill="var(--ink)" font-family="sans-serif" font-weight="700">${esc(p.m.id)}</text>`;
    }
    document.getElementById("paretoSvg").innerHTML = svg;
  }

  function renderHeatmap() {
    const langs = PAYLOAD.langs || [];
    const mat = (PAYLOAD.per_lang || {})[heatMetric] || {};
    const lowerBetter = heatMetric === "fert";
    document.getElementById("heatHeadRow").innerHTML =
      `<th>Model</th>` + langs.map((l) => `<th class="center"><code>${esc(l)}</code></th>`).join("");
    document.getElementById("heatBody").innerHTML = (PAYLOAD.models || []).map((m) => {
      const vals = mat[m.id] || langs.map(() => 0);
      return `<tr><td><strong>${esc(m.id)}</strong></td>` + vals.map((v) => {
        const norm = lowerBetter
          ? Math.max(0, Math.min(1, 1 - (v - 1.5) / 2.2))
          : Math.max(0, Math.min(1, v / 85));
        const bg = norm > 0.65 ? "rgba(30,92,58,0.20)" : (norm < 0.30 ? "rgba(140,29,36,0.16)" : "transparent");
        const txt = lowerBetter ? v.toFixed(2) : v.toFixed(1);
        return `<td class="heat" style="background:${bg}">${txt}</td>`;
      }).join("") + `</tr>`;
    }).join("");
  }

  function renderShatter() {
    const lang = document.getElementById("shatterLang").value;
    const entry = (PAYLOAD.shatter_samples || {})[lang];
    const el = document.getElementById("shatterContainer");
    if (!entry) {
      el.innerHTML = "<p>No shatter sample available.</p>";
      return;
    }
    const mkChips = (arr) => arr.map((t, i) => `<span class="tchip ${i % 2 ? "alt" : ""}">${esc(t)}</span>`).join("");
    let html = `
      <div style="margin-bottom:10px">
        <div style="font-family:var(--sans);font-size:0.74rem;font-weight:700">DabaX Linguistic Morphemes (${entry.oracle.length})</div>
        <div class="token-chips">${mkChips(entry.oracle)}</div>
      </div>`;
    const tokEntries = Object.entries(entry.tokenizers || {}).slice(0, 6);
    for (const [mid, pieces] of tokEntries) {
      html += `
        <div style="margin-bottom:10px">
          <div style="font-family:var(--sans);font-size:0.74rem;font-weight:700">${esc(mid)} (${pieces.length} subwords)</div>
          <div class="token-chips">${mkChips(pieces)}</div>
        </div>`;
    }
    el.innerHTML = html;
  }

  function renderDiacBars() {
    const models = PAYLOAD.models || [];
    const maxVal = Math.max(3.8, ...models.map((m) => Math.max(m.ascii_fert || 1, m.ext_fert || 1)));
    document.getElementById("diacBars").innerHTML = models.map((m) => `
      <div>
        <div style="display:flex;justify-content:space-between;font-family:var(--sans);font-size:0.76rem;font-weight:700">
          <span>${esc(m.id)} <small style="font-weight:400;color:var(--ink-soft)">(${esc(m.tokenizer_mode)})</small></span>
          <span style="font-family:var(--mono)">ASCII ${m.ascii_fert.toFixed(2)} vs. Ext ${m.ext_fert.toFixed(2)} (${m.diac.toFixed(2)}&times;)</span>
        </div>
        <div class="bar-track" style="margin-bottom:3px"><div class="bar-fill" style="width:${(m.ascii_fert / maxVal) * 100}%;background:var(--forest)"></div></div>
        <div class="bar-track"><div class="bar-fill" style="width:${(m.ext_fert / maxVal) * 100}%;background:var(--crimson)"></div></div>
      </div>
    `).join("");
  }

  function renderLid() {
    const langs = PAYLOAD.langs || [];
    const select = document.getElementById("lidModelSelect");
    const chosen = select ? select.value : lidModel;
    const mat = (PAYLOAD.lid_confusion || {})[chosen];
    if (!mat) return;
    document.getElementById("lidHead").innerHTML =
      `<tr><th>True \\ Pred</th>` + langs.map((l) => `<th class="center"><code>${esc(l)}</code></th>`).join("") + `</tr>`;
    document.getElementById("lidBody").innerHTML = langs.map((rl, r) =>
      `<tr><th><code>${esc(rl)}</code></th>` + langs.map((_cl, c) => {
        const val = mat[r][c] || 0;
        const bg = r === c ? "rgba(30,92,58,0.22)" : (val >= 2 ? "rgba(140,29,36,0.18)" : "transparent");
        return `<td class="heat" style="background:${bg}">${val ? val : "&middot;"}</td>`;
      }).join("") + `</tr>`
    ).join("");
  }

  function renderInspector(lang) {
    inspLang = lang;
    const wrap = document.getElementById("inspectorContainer");
    if (!SAMPLES || !SAMPLES[lang]) {
      wrap.innerHTML = "<p>Sample completions not yet published.</p>";
      return;
    }
    const d = SAMPLES[lang];
    wrap.innerHTML = `
      <div style="margin-bottom:10px;padding:10px 12px;background:var(--paper-card);border:1px solid var(--rule-heavy);font-size:0.9rem">
        <strong>Input Sentence (<code>${esc(d.lang)}</code> &middot; ${esc(d.lang_name)} &middot; <code>${esc(d.sample_id)}</code>):</strong>
        <div style="margin-top:4px;font-family:var(--mono);font-size:0.86rem">${esc(d.prompt)}</div>
      </div>
      <div class="inspect-grid">
        <div class="inspect-card oracle">
          <h4>DabaX(strict_parse=True) Oracle</h4>
          <pre class="snippet">${esc(d.oracle)}</pre>
        </div>
        ${(d.models || []).map((m) => `
          <div class="inspect-card ${esc(m.trackCls)}">
            <h4>${esc(m.id)} <span class="tag ${esc(m.trackCls)}">${esc(m.track)}</span></h4>
            <pre class="snippet">${esc(m.output)}</pre>
          </div>
        `).join("")}
      </div>
    `;
  }

  async function init() {
    PAYLOAD = await fetchJson("data/leaderboard.json");
    try {
      SAMPLES = await fetchJson("data/samples.json");
    } catch (_e) {
      SAMPLES = {};
    }

    const models = PAYLOAD.models || [];
    const statusEl = document.getElementById("lbLiveBanner");
    if (statusEl) {
      statusEl.textContent = `LIVE MALI13 (${PAYLOAD.tier}) • ${models.length} MODELS • ${PAYLOAD.total_samples} SAMPLES • UPDATED ${PAYLOAD.generated_at} • MANIFEST SHA ${PAYLOAD.manifest_sha256}`;
    }

    const pendingEl = document.getElementById("pendingModelsNote");
    if (pendingEl && (PAYLOAD.pending_models || []).length > 0) {
      pendingEl.style.display = "block";
      pendingEl.innerHTML =
        "<strong>Pending / Unscored Models:</strong> " +
        PAYLOAD.pending_models.map((p) => `<code>${esc(p.id)}</code> (${esc(p.provider)})`).join(", ");
    }

    const allBtn = document.querySelector('#backendFilter button[data-filter="ALL"]');
    if (allBtn) allBtn.textContent = `All Models (${models.length})`;

    const lidSelect = document.getElementById("lidModelSelect");
    if (lidSelect) {
      lidSelect.innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)} (LID ${m.lid.toFixed(1)})</option>`).join("");
      lidModel = models[0] ? models[0].id : null;
      lidSelect.addEventListener("change", renderLid);
    }

    document.getElementById("backendFilter").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-filter]");
      if (!b) return;
      lbFilter = b.dataset.filter;
      document.querySelectorAll("#backendFilter button").forEach((x) => x.classList.toggle("active", x === b));
      renderLeaderboard();
    });

    document.querySelectorAll("#lbTable th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.dataset.sort;
        if (lbSort === col) lbAsc = !lbAsc;
        else {
          lbSort = col;
          lbAsc = col === "id" || col === "fert" || col === "diac" || col === "bpb" || col === "lat";
        }
        renderLeaderboard();
      });
    });

    document.getElementById("paretoAxisBtns").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-axis]");
      if (!b) return;
      paretoAxis = b.dataset.axis;
      document.querySelectorAll("#paretoAxisBtns button").forEach((x) => x.classList.toggle("active", x === b));
      renderPareto();
    });

    document.getElementById("heatMetricBtns").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-hm]");
      if (!b) return;
      heatMetric = b.dataset.hm;
      document.querySelectorAll("#heatMetricBtns button").forEach((x) => x.classList.toggle("active", x === b));
      renderHeatmap();
    });

    document.getElementById("shatterLang").addEventListener("change", renderShatter);

    document.getElementById("inspLangBtns").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-lang]");
      if (!b) return;
      document.querySelectorAll("#inspLangBtns button").forEach((x) => x.classList.toggle("active", x === b));
      renderInspector(b.dataset.lang);
    });

    renderLeaderboard();
    renderPareto();
    renderHeatmap();
    renderShatter();
    renderDiacBars();
    renderLid();
    renderInspector(inspLang);
  }

  document.addEventListener("DOMContentLoaded", () => {
    init().catch((err) => {
      const el = document.getElementById("lbBody");
      if (el) el.innerHTML = `<tr><td colspan="14">Failed to load live leaderboard JSON: ${esc(err.message)}</td></tr>`;
    });
  });
})();
