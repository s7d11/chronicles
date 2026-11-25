/**
 * Interactive Edition No. 2 broadsheet controller:
 * - Renders the self-organized Topology Zoo (SVG) + Consensus & Shadow-Patch Ledger
 * - Drives the Per-Cluster Exchange, Shadow DabaX & Native-Command Inspector
 */

(function () {
  "use strict";

  const esc = (s) => window.Chronicle.esc(s);

  function renderTopoSvg(plan) {
    const n = plan.n_agents || 2;
    const topo = plan.topology || "star";
    const w = 132;
    const h = 84;
    const cx = w / 2;
    const cy = h / 2;
    const pts = [];

    if (topo === "star") {
      pts.push([cx, cy]);
      for (let i = 1; i < n; i++) {
        const a = ((i - 1) / Math.max(1, n - 1)) * Math.PI * 2 - Math.PI / 2;
        pts.push([cx + Math.cos(a) * 34, cy + Math.sin(a) * 28]);
      }
    } else if (topo === "pipeline") {
      for (let i = 0; i < n; i++) {
        pts.push([18 + (i * (w - 36)) / Math.max(1, n - 1), cy + (i % 2 === 0 ? -9 : 9)]);
      }
    } else if (topo === "hierarchical") {
      pts.push([cx, 16]);
      for (let i = 1; i < n; i++) {
        pts.push([20 + ((i - 1) * (w - 40)) / Math.max(1, n - 1), h - 18]);
      }
    } else {
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 - Math.PI / 2;
        pts.push([cx + Math.cos(a) * 36, cy + Math.sin(a) * 26]);
      }
    }

    const edges = (plan.edges || [])
      .map(([u, v]) => {
        const ui = Number(String(u).replace(/\D/g, "")) - 1;
        const vi = Number(String(v).replace(/\D/g, "")) - 1;
        if (!pts[ui] || !pts[vi]) return "";
        return `<line x1="${pts[ui][0].toFixed(1)}" y1="${pts[ui][1].toFixed(1)}" x2="${pts[vi][0].toFixed(1)}" y2="${pts[vi][1].toFixed(1)}" stroke="#8a8275" stroke-width="1.2" />`;
      })
      .join("");

    const nodes = pts
      .map(
        ([x, y], i) =>
          `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${i === 0 ? "#9a6314" : "#1b4d6e"}" stroke="#fbf8f0" stroke-width="1.3" />` +
          `<text x="${x.toFixed(1)}" y="${(y + 2.5).toFixed(1)}" text-anchor="middle" fill="#fbf8f0" font-size="6.5" font-family="monospace">a${i + 1}</text>`
      )
      .join("");

    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" style="display:block;background:var(--paper);border:1px solid var(--rule-hair)">${edges}${nodes}</svg>`;
  }

  async function initEdition2() {
    let data = null;
    const mount = document.getElementById("swarmTheaterMount");
    const liveOrigin = mount && mount.dataset.liveOrigin ? mount.dataset.liveOrigin.replace(/\/$/, "") : "";
    const wantLive = new URLSearchParams(window.location.search).get("live") === "1";
    const isSameLiveOrigin = Boolean(liveOrigin && window.location.origin.startsWith(liveOrigin));
    const isLocalHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    const candidates = [];
    if (isSameLiveOrigin || isLocalHost) {
      candidates.push("/api/edition2.json");
    }
    if (wantLive && liveOrigin && !isSameLiveOrigin) {
      candidates.push(`${liveOrigin}/api/edition2.json`);
    }
    candidates.push("../../data/edition2.json");
    if (!wantLive && liveOrigin && !isSameLiveOrigin) {
      candidates.push(`${liveOrigin}/api/edition2.json`);
    }

    for (const url of candidates) {
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 3500);
        const resp = await fetch(url, { cache: "no-store", signal: ctrl.signal });
        clearTimeout(timer);
        if (resp.ok) {
          const parsed = await resp.json();
          if (parsed && Array.isArray(parsed.langs) && parsed.langs.length > 0) {
            data = parsed;
            break;
          }
        }
      } catch (_) {}
    }

    if (!data) {
      document.getElementById("ed2LedgerBody").innerHTML =
        `<tr><td colspan="10">Awaiting published <code>data/edition2.json</code>.</td></tr>`;
      return;
    }

    const langs = data.langs || [];
    const lbRows = data.leaderboard || [];
    const metaEl = document.getElementById("ed2RunMeta");
    if (metaEl) {
      metaEl.textContent = `EDITION NO. 2 • ${langs.length} LANGUAGES (${langs.join(", ").toUpperCase()}) • COORDINATOR: ${data.coordinator_model} • PUBLISHED ${data.generated_at}`;
    }

    // Aggregate Ledger Table
    document.getElementById("ed2LedgerBody").innerHTML = lbRows
      .map(
        (r, idx) => `
      <tr>
        <td class="num">${idx + 1}</td>
        <td><strong>${esc(r.model_name)}</strong></td>
        <td class="num">${r.avg_agents.toFixed(1)}</td>
        <td><span class="tag slm">${esc(r.dominant_topology)}</span></td>
        <td class="num">${r.unanimous_rate_pct.toFixed(0)}% (${r.avg_rounds.toFixed(1)}R)</td>
        <td class="num">+${r.total_dict_added} (${r.compile_ok_rate_pct.toFixed(0)}% OK)</td>
        <td class="num">${r.avg_test_delta_pct >= 0 ? "+" : ""}${r.avg_test_delta_pct.toFixed(2)}%</td>
        <td class="num">${r.avg_native_parse_pct.toFixed(1)}%</td>
        <td class="num">${r.avg_header_compliance_pct.toFixed(0)}%</td>
        <td class="num"><strong>${r.avg_coordinator_score.toFixed(1)}</strong></td>
      </tr>`
      )
      .join("");

    // Populate Language & Cluster Inspector
    const langSel = document.getElementById("ed2InspLang");
    const clusSel = document.getElementById("ed2InspCluster");
    langSel.innerHTML = langs
      .map((l) => `<option value="${esc(l)}">${esc(l.toUpperCase())} — ${esc(data.languages[l].lang_name)}</option>`)
      .join("");

    function renderInspector() {
      const lang = langSel.value || langs[0];
      const ldoc = data.languages[lang];
      if (!ldoc) return;
      const clusters = ldoc.clusters || [];
      if (!clusSel.options.length || clusSel.dataset.lang !== lang) {
        const prev = clusSel.value;
        clusSel.dataset.lang = lang;
        clusSel.innerHTML = clusters
          .map((c) => `<option value="${esc(c.cluster_slug)}">${esc(c.model_name)} (${c.plan.n_agents}x ${c.plan.topology})</option>`)
          .join("");
        if ([...clusSel.options].some((o) => o.value === prev)) clusSel.value = prev;
      }
      const c = clusters.find((x) => x.cluster_slug === clusSel.value) || clusters[0];
      if (!c) return;

      // Topology Zoo Strip for selected language
      document.getElementById("ed2ZooGrid").innerHTML = clusters
        .map(
          (item) => `
        <div class="flow-card ${item.cluster_slug === c.cluster_slug ? "note" : ""}" style="padding:8px;cursor:pointer" data-pick="${esc(item.cluster_slug)}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;font-size:0.78rem">
            <strong>${esc(item.model_name)}</strong>
            <span class="tag ${item.unanimous ? "sebeni" : "api"}">${item.plan.n_agents}&times; ${esc(item.plan.topology)}</span>
          </div>
          ${renderTopoSvg(item.plan)}
          <div style="font-family:var(--mono);font-size:0.68rem;color:var(--ink-soft);margin-top:5px">
            Agree ${item.agreement_pct.toFixed(0)}% (${item.rounds_used}R) · Shadow ${item.shadow.test_delta_pct >= 0 ? "+" : ""}${item.shadow.test_delta_pct.toFixed(1)}% · Score ${item.coordinator_score.toFixed(1)}
          </div>
        </div>`
        )
        .join("");

      document.querySelectorAll("#ed2ZooGrid [data-pick]").forEach((card) => {
        card.addEventListener("click", () => {
          clusSel.value = card.dataset.pick;
          renderInspector();
        });
      });

      const directiveBox = document.getElementById("ed2DirectiveBox");
      directiveBox.innerHTML = `
        <strong>Coordinator Brief (Nemotron-3-Ultra-550B):</strong> ${esc(ldoc.coordinator_brief)}<br/>
        <strong>Randomized <code>prompt_native.csv</code> Directive:</strong> <code>${esc(ldoc.native_command)}</code><br/>
        <strong>Coordinator Verdict for ${esc(ldoc.lang_name)}:</strong> ${esc(ldoc.coordinator_verdict)}
      `;

      const exHtml = (c.exchanges || [])
        .map(
          (e) =>
            `<div class="ex-card"><div class="ex-meta"><span>R${e.round_num} · ${esc(e.src)} &rarr; ${esc(e.dst)}</span><span>${e.agree === true ? "AGREE" : e.agree === false ? "REVISE" : "PROPOSE"}</span></div><div class="ex-body">${esc(e.summary)}</div></div>`
        )
        .join("");

      const dictHtml = (c.dict_deltas || [])
        .map((d) => `<li><code>\\lx ${esc(d.lx)}</code> (<code>${esc(d.ps)}</code>): ${esc(d.ge)}</li>`)
        .join("") || "<li>No novel dictionary deltas proposed beyond baseline.</li>";

      document.getElementById("ed2DetailPane").innerHTML = `
        <div class="two-col">
          <div class="factbox" style="margin:0">
            <h3>${esc(c.model_name)} &mdash; ${c.plan.n_agents} Agents (${esc(c.plan.topology)})</h3>
            <p style="font-size:0.82rem;margin:0 0 8px"><em>${esc(c.plan.rationale)}</em></p>
            <p style="font-size:0.8rem;margin:0 0 8px"><strong>Coordinator Critique (${c.coordinator_score.toFixed(1)}):</strong> ${esc(c.coordinator_note)}</p>
            <div style="font-family:var(--sans);font-size:0.74rem;font-weight:700;text-transform:uppercase;margin:8px 0 4px">Cluster Exchanges (${c.n_exchanges})</div>
            <div style="max-height:240px;overflow-y:auto;display:grid;gap:5px">${exHtml}</div>
          </div>
          <div class="factbox" style="margin:0;border-top-color:var(--crimson)">
            <h3>Shadow DabaX (+${c.shadow.dict_entries_added} dict) &amp; Native Generation</h3>
            <p style="font-size:0.8rem;margin:0 0 6px">
              <strong>Compile:</strong> ${c.shadow.dict_compile_ok ? "PASS (0.0% phantom)" : "ERR"} &bull;
              <strong>test.json delta:</strong> ${c.shadow.baseline_test_parse_pct}% &rarr; ${c.shadow.shadow_test_parse_pct}% (${c.shadow.test_delta_pct >= 0 ? "+" : ""}${c.shadow.test_delta_pct}%) &bull;
              <strong>dev-mini delta:</strong> ${c.shadow.devmini_delta_pct >= 0 ? "+" : ""}${c.shadow.devmini_delta_pct}%
            </p>
            <ul style="font-size:0.78rem;margin:0 0 10px;padding-left:18px">${dictHtml}</ul>
            <div style="font-family:var(--sans);font-size:0.74rem;font-weight:700;text-transform:uppercase;margin-bottom:4px">
              Native Output (${c.native_gen.shadow_parse_pct}% DabaX · ${c.native_gen.header_compliance}% Headers · ${c.native_gen.leakage_pct}% Leakage)
            </div>
            <pre style="white-space:pre-wrap;font-family:var(--mono);font-size:0.76rem;background:var(--paper);border:1px solid var(--rule-hair);padding:8px;max-height:200px;overflow-y:auto;margin:0">${esc(c.native_gen.output_text)}</pre>
          </div>
        </div>
      `;
    }

    langSel.addEventListener("change", renderInspector);
    clusSel.addEventListener("change", renderInspector);
    renderInspector();
  }

  document.addEventListener("DOMContentLoaded", initEdition2);
})();
