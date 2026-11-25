(function () {
  const THEME_KEY = "sebeni_chronicle_theme";

  function applyTheme(theme) {
    const t = theme === "evening" ? "evening" : "newsprint";
    document.documentElement.dataset.theme = t;
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.textContent = t === "evening" ? "Morning Newsprint" : "Evening Edition";
    }
  }

  function initThemeToggle() {
    try {
      applyTheme(localStorage.getItem(THEME_KEY) || "newsprint");
    } catch (_e) {
      applyTheme("newsprint");
    }
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "evening" ? "newsprint" : "evening";
      applyTheme(next);
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch (_e) {}
    });
  }

  function esc(val) {
    return String(val ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status + " loading " + url);
    return res.json();
  }

  async function hydrateFrontPage() {
    const kpiBox = document.getElementById("liveFactbox");
    const ed2Box = document.getElementById("liveEd2Factbox");
    const ed2Body = document.getElementById("frontEd2Body");
    const teaserBody = document.getElementById("frontTeaserBody");
    const archiveBody = document.getElementById("editionsArchiveBody");
    const metaEl = document.getElementById("liveRunMeta");
    const ribbonParts = [];

    if (ed2Box || ed2Body) {
      try {
        const ed2 = await fetchJson("data/edition2.json");
        const langs = ed2.langs || [];
        const lb = ed2.leaderboard || [];
        const topSwarm = lb[0];
        const totalSessions = langs.reduce(
          (sum, l) => sum + ((ed2.languages?.[l]?.clusters || []).length || 0),
          0
        );
        const totalDictAdded = lb.reduce((sum, r) => sum + (r.total_dict_added || 0), 0);
        const cleanCompiles = langs.reduce(
          (sum, l) =>
            sum +
            (ed2.languages?.[l]?.clusters || []).filter((c) => c.shadow?.dict_compile_ok).length,
          0
        );
        const topNative = [...lb].sort((a, b) => b.avg_native_parse_pct - a.avg_native_parse_pct)[0];
        ribbonParts.push(
          `EDITION NO. 2: ${langs.length}/13 LANGUAGES (${totalSessions} SESSIONS, +${totalDictAdded} VERIFIED DICT ENTRIES)`
        );

        if (ed2Box && topSwarm) {
          ed2Box.innerHTML = `
            <div class="spec-item">
              <div class="lbl">Edition No. 2 Coverage</div>
              <div class="val">${langs.length}/13 MULTI13 &bull; ${totalSessions} Sessions</div>
              <div class="sub">9 Adaption clusters &bull; Coordinated by <code>nemotron-3-ultra-550b</code></div>
            </div>
            <div class="spec-item">
              <div class="lbl">Top Swarm Cluster</div>
              <div class="val">${esc(topSwarm.model_name)} (${topSwarm.avg_coordinator_score.toFixed(1)})</div>
              <div class="sub">${topSwarm.unanimous_rate_pct.toFixed(0)}% unanimity &bull; ${esc(topSwarm.dominant_topology)} (${topSwarm.avg_agents.toFixed(1)} agents)</div>
            </div>
            <div class="spec-item">
              <div class="lbl">Shadow DabaX Integrity</div>
              <div class="val">${cleanCompiles}/${totalSessions} Clean &bull; +${totalDictAdded} Entries</div>
              <div class="sub">0.0% phantom inflation &bull; Strictly validated parses</div>
            </div>
            <div class="spec-item">
              <div class="lbl">Native-Command Parse Leader</div>
              <div class="val">${esc(topNative.model_name)} (${topNative.avg_native_parse_pct.toFixed(1)}%)</div>
              <div class="sub">${topNative.avg_header_compliance_pct.toFixed(1)}% header compliance on <code>prompt_native.csv</code></div>
            </div>
          `;
        }

        if (ed2Body) {
          ed2Body.innerHTML = lb.map((r, idx) => `
            <tr>
              <td class="num">${idx + 1}</td>
              <td><strong>${esc(r.model_name)}</strong></td>
              <td class="num">${r.avg_agents.toFixed(1)}</td>
              <td><span class="tag slm">${esc(r.dominant_topology)}</span></td>
              <td class="num">${r.unanimous_rate_pct.toFixed(0)}% (${r.avg_rounds.toFixed(1)}R)</td>
              <td class="num">+${r.total_dict_added} (${r.compile_ok_rate_pct.toFixed(0)}% OK)</td>
              <td class="num">${r.avg_test_delta_pct >= 0 ? "+" : ""}${r.avg_test_delta_pct.toFixed(2)}%</td>
              <td class="num">${r.avg_native_parse_pct.toFixed(1)}%</td>
              <td class="num"><strong>${r.avg_coordinator_score.toFixed(1)}</strong></td>
            </tr>
          `).join("");
        }
      } catch (err) {
        if (ed2Box) {
          ed2Box.innerHTML = `<div class="spec-item"><div class="lbl">Edition No. 2 Status</div><div class="sub">Awaiting edition2.json (${esc(err.message)})</div></div>`;
        }
      }
    }

    if (kpiBox || teaserBody) {
      try {
        const data = await fetchJson("data/leaderboard.json");
        const models = data.models || [];
        const sortedQ = [...models].sort((a, b) => b.q - a.q);
        const topQ = sortedQ[0];
        const topMin3 = [...models].sort((a, b) => b.q_min3 - a.q_min3)[0];
        const bestFert = [...models].sort((a, b) => a.fert - b.fert)[0];
        ribbonParts.push(`MALI13 LIVE: ${models.length} MODELS SCORED (${data.total_samples} SAMPLES)`);

        if (kpiBox && topQ) {
          kpiBox.innerHTML = `
            <div class="spec-item">
              <div class="lbl">Models Benchmarked Live</div>
              <div class="val">${models.length} Live &bull; ${data.total_samples} Samples</div>
              <div class="sub">13 <code>MULTI13</code> languages &bull; 5 tasks &bull; <code>dev-mini</code></div>
            </div>
            <div class="spec-item">
              <div class="lbl">Overall MALI13-Q Leader</div>
              <div class="val">${esc(topQ.id)} (${topQ.q.toFixed(1)})</div>
              <div class="sub">Morph ${topQ.morph.toFixed(1)} &bull; Gen ${topQ.gen.toFixed(1)} &bull; LID ${topQ.lid.toFixed(1)} &bull; Cloze ${topQ.cloze.toFixed(1)}</div>
            </div>
            <div class="spec-item">
              <div class="lbl">Cross-Language Equity Leader (Q_min3)</div>
              <div class="val">${esc(topMin3.id)} (${topMin3.q_min3.toFixed(1)})</div>
              <div class="sub">Worst-3-language mean &bull; Spread Q_gap ${topMin3.q_gap.toFixed(1)}</div>
            </div>
            <div class="spec-item">
              <div class="lbl">Best Subword Compression</div>
              <div class="val">${esc(bestFert.id)} (${bestFert.fert.toFixed(2)} tok/w)</div>
              <div class="sub">Diacritic Penalty ${bestFert.diac.toFixed(2)}&times;</div>
            </div>
          `;
        }

        if (teaserBody) {
          teaserBody.innerHTML = sortedQ.slice(0, 8).map((m, idx) => `
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
            </tr>
          `).join("");
        }
      } catch (err) {
        if (kpiBox) {
          kpiBox.innerHTML = `<div class="spec-item"><div class="lbl">Live Status</div><div class="sub">Awaiting leaderboard.json (${esc(err.message)})</div></div>`;
        }
      }
    }

    if (metaEl && ribbonParts.length) {
      metaEl.textContent = `THE SEBENI CHRONICLE • ${ribbonParts.join(" • ")}`;
    }

    if (archiveBody) {
      try {
        const editions = await fetchJson("data/editions.json");
        archiveBody.innerHTML = editions.map((ed) => `
          <tr>
            <td><code>${esc(ed.vol_issue)}</code></td>
            <td>${esc(ed.date)}</td>
            <td><strong><a href="${esc(ed.url)}">${esc(ed.title)}</a></strong><br/><small style="color:var(--ink-muted)">${esc(ed.summary)}</small></td>
            <td><span class="tag ${esc(ed.tagCls || "sebeni")}">${esc(ed.kind)}</span></td>
          </tr>
        `).join("");
      } catch (_e) {}
    }
  }

  window.Chronicle = { esc, fetchJson };
  document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    hydrateFrontPage();
  });
})();
