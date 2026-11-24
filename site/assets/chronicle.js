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
    const teaserBody = document.getElementById("frontTeaserBody");
    const archiveBody = document.getElementById("editionsArchiveBody");

    if (kpiBox || teaserBody) {
      try {
        const data = await fetchJson("data/leaderboard.json");
        const models = data.models || [];
        const sortedQ = [...models].sort((a, b) => b.q - a.q);
        const topQ = sortedQ[0];
        const topMin3 = [...models].sort((a, b) => b.q_min3 - a.q_min3)[0];
        const bestFert = [...models].sort((a, b) => a.fert - b.fert)[0];

        const metaEl = document.getElementById("liveRunMeta");
        if (metaEl) {
          metaEl.textContent = `Updated ${data.generated_at} • ${models.length} models scored on ${data.total_samples} dev-mini samples (SHA ${data.manifest_sha256})`;
        }

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
