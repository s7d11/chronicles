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
    if (!btn || btn.dataset.themeBound) return;
    btn.dataset.themeBound = "1";
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

  function getSiteRoot() {
    const raw = (document.body && document.body.dataset && document.body.dataset.root) || ".";
    return raw.replace(/\/+$/, "");
  }

  function resolveSiteUrl(relPath) {
    const clean = String(relPath || "").replace(/^\/+/, "");
    const root = getSiteRoot();
    if (!root || root === ".") return clean;
    return `${root}/${clean}`;
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status + " loading " + url);
    return res.json();
  }

  async function hydrateFrontPage() {
    const kpiBox = document.getElementById("liveFactbox");
    const vol2Box = document.getElementById("liveVol2Factbox");
    const ed4Box = document.getElementById("liveEd4Factbox");
    const ed3Box = document.getElementById("liveEd3Factbox");
    const ed3Body = document.getElementById("frontEd3Body");
    const ed2Box = document.getElementById("liveEd2Factbox");
    const ed2Body = document.getElementById("frontEd2Body");
    const teaserBody = document.getElementById("frontTeaserBody");
    const archiveBody = document.getElementById("editionsArchiveBody");
    const metaEl = document.getElementById("liveRunMeta");
    const ribbonParts = [];

    if (vol2Box) {
      try {
        const v2 = await fetchJson("data/vol2_meta.json");
        ribbonParts.push(
          `VOL. II NO. 1: ${Number(v2.total_verified_incidents || 0).toLocaleString()} MALI EVENTS · ${v2.total_hypotheses || 0} MATH PROOFS · AUROC ${v2.global_auroc ?? "-"}`
        );
        vol2Box.innerHTML = `
          <div class="spec-item">
            <div class="lbl">Verified Incidents &amp; Clusters</div>
            <div class="val">${Number(v2.total_verified_incidents || 0).toLocaleString()} / ${v2.total_clusters || 0} clusters</div>
            <div class="sub">From ${Number(v2.total_raw_records || 0).toLocaleString()} raw records (${Object.keys(v2.perspective_counts || {}).length} perspectives)</div>
          </div>
          <div class="spec-item">
            <div class="lbl">Mathematical Proof Engine</div>
            <div class="val">${v2.proven_or_supported_count || 0} / ${v2.total_hypotheses || 0} Verified</div>
            <div class="sub">Hawkes &middot; Overdispersion &middot; Markov &middot; Harmattan &middot; Beta-Binomial</div>
          </div>
          <div class="spec-item">
            <div class="lbl">Strict Walk-Forward Score</div>
            <div class="val">AUROC ${v2.global_auroc ?? "-"} (Brier ${v2.global_brier ?? "-"})</div>
            <div class="sub">${v2.walk_forward_folds || 0} causal folds &bull; NVIDIA + OpenRouter Free + Gemini 3.1 Flash-Lite</div>
          </div>
        `;
      } catch (err) {
        vol2Box.innerHTML = `<div class="spec-item"><div class="lbl">Vol. II Observatory</div><div class="sub">Awaiting vol2_meta.json (${esc(err.message)})</div></div>`;
      }
    }

    if (ed4Box) {
      try {
        const ed4 = await fetchJson("data/edition4.json");
        const bam = (ed4.languages && ed4.languages[ed4.launch_lang || "bam"]) || {};
        const baseLx = Number(bam.baseline_lx_count || 0);
        const addedLx = Number(bam.added_lx_count || 0);
        const addedVa = Number(bam.added_va_count || 0);
        const addedGram = Number(bam.added_gram_rules_count || 0);
        ribbonParts.push(
          `EDITION NO. 4: ${esc(bam.lang_name || "bam")} ${(baseLx + addedLx).toLocaleString()} / 50,000 (+${addedLx} lx, +${addedVa} va, +${addedGram} gram)`
        );
        ed4Box.innerHTML = `
          <div class="spec-item">
            <div class="lbl">Launch Village (${esc(ed4.launch_lang || "bam")})</div>
            <div class="val">${(baseLx + addedLx).toLocaleString()} / 50,000</div>
            <div class="sub">Baseline ${baseLx.toLocaleString()} + ${addedLx} \\lx + ${addedVa} \\va + ${addedGram} .gram</div>
          </div>
          <div class="spec-item">
            <div class="lbl">Orchestrator &amp; Critic</div>
            <div class="val">Mande &times; Gemini 3.5 Flash</div>
            <div class="sub"><code>nemotron-3-ultra-550b-a55b</code> + <code>gemini-3.5-flash</code> + 9 Adaption models</div>
          </div>
          <div class="spec-item">
            <div class="lbl">3D Simulation Events</div>
            <div class="val">${ed4.total_events || 0} Events</div>
            <div class="sub">${(bam.active_agents || []).length || 11} village &amp; Council agents (${bam.total_turns || 0} turns)</div>
          </div>
        `;
      } catch (err) {
        ed4Box.innerHTML = `<div class="spec-item"><div class="lbl">Edition No. 4 Village</div><div class="sub">Awaiting edition4.json (${esc(err.message)})</div></div>`;
      }
    }

    if (ed3Box || ed3Body) {
      try {
        let ed3;
        try {
          ed3 = await fetchJson("data/edition3.json");
        } catch (_staticErr) {
          ed3 = await fetchJson("https://sebeni-swarm-live-6orlk3ix5a-uk.a.run.app/api/ed3/edition3.json");
        }
        const g = ed3.global_metrics || {};
        const lb3 = ed3.model_leaderboard || [];
        const top3 = lb3[0];
        ribbonParts.push(
          `EDITION NO. 3: ${g.n_langs || (ed3.langs || []).length}/14 CODES (${ed3.total_sentences || 0} SENTENCES, d(MER)=${g.mean_cohens_d_mer_corrupt ?? "-"})`
        );

        if (ed3Box) {
          ed3Box.innerHTML = `
            <div class="spec-item">
              <div class="lbl">Edition No. 3 Stress Scope</div>
              <div class="val">${g.n_langs || 0}/14 Codes &bull; ${ed3.total_sentences || 0} Rows</div>
              <div class="sub">13 <code>MALI13</code> languages + <code>bbo</code> OOD control (<code>dataset_300_samples.jsonl</code>)</div>
            </div>
            <div class="spec-item">
              <div class="lbl">MER &amp; MCS Effectiveness</div>
              <div class="val">d=${g.mean_cohens_d_mer_corrupt ?? "-"} &bull; +${g.mean_shift_mcs_recovery_pct ?? "-"} pp LCS</div>
              <div class="sub">${g.mean_mer_saturation_prevented_pct ?? "-"}% overgen saturation caught by <code>bounded_mer</code></div>
            </div>
            <div class="spec-item">
              <div class="lbl">UWEC OOD Discriminability</div>
              <div class="val">d=${g.mean_cohens_d_uwec_ood ?? "-"} &bull; ${g.mean_ood_strict_gap_pct ?? "-"} pp Gap</div>
              <div class="sub"><code>bbo</code> OOD strict parse ${g.ood_bbo_strict_parse_pct ?? "-"}% vs. MALI13 ${g.mean_clean_strict_parse_pct ?? "-"}%</div>
            </div>
            ${top3 ? `<div class="spec-item">
              <div class="lbl">Top NVIDIA Deduction Panelist</div>
              <div class="val">${esc(top3.model_name)} (${top3.mean_effectiveness_score.toFixed(1)}/100)</div>
              <div class="sub">${top3.oracle_verified}/${top3.oracle_total} live DabaX probes verified</div>
            </div>` : ""}
          `;
        }

        if (ed3Body) {
          ed3Body.innerHTML = lb3.map((r, idx) => `
            <tr>
              <td class="num">${idx + 1}</td>
              <td><strong>${esc(r.model_name)}</strong></td>
              <td>${esc(r.params_label)}</td>
              <td class="num">${r.langs_completed}/14</td>
              <td class="num">${r.oracle_verified}/${r.oracle_total} (${r.oracle_pass_pct.toFixed(0)}%)</td>
              <td class="num">${r.mean_r1_latency_s.toFixed(1)}s / ${r.mean_r2_latency_s.toFixed(1)}s</td>
              <td><span class="tag ${r.dominant_verdict === "PROVEN_EFFECTIVE" ? "sebeni" : "slm"}">${esc(r.dominant_verdict)}</span></td>
              <td class="num"><strong>${r.mean_effectiveness_score.toFixed(1)}</strong></td>
            </tr>
          `).join("");
        }
      } catch (err) {
        if (ed3Box) {
          ed3Box.innerHTML = `<div class="spec-item"><div class="lbl">Edition No. 3 Status</div><div class="sub">Awaiting edition3.json (${esc(err.message)})</div></div>`;
        }
      }
    }

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
            <td class="num">${esc(ed.date)}</td>
            <td><strong><a href="${esc(ed.url)}">${esc(ed.title)}</a></strong><div style="font-size:0.8125rem;color:var(--ink-muted);margin-top:2px">${esc(ed.summary)}</div></td>
            <td><span class="tag ${esc(ed.tagCls || "sebeni")}">${esc(ed.kind)}</span></td>
          </tr>
        `).join("");
      } catch (_e) {}
    }
  }

  async function hydrateCrossEditionNav() {
    const pageId = (document.body && document.body.dataset && document.body.dataset.pageId) || "";
    const navEl = document.querySelector("nav.ed-nav[data-chronicle-nav]");
    if (!navEl && !pageId) return;

    let editions = [];
    try {
      editions = await fetchJson(resolveSiteUrl("data/editions.json"));
    } catch (_e) {
      return;
    }
    if (!Array.isArray(editions) || !editions.length) return;

    const byId = {};
    const referencedBy = {};
    editions.forEach((item) => {
      if (!item || !item.id) return;
      byId[item.id] = item;
      referencedBy[item.id] = [];
    });

    editions.forEach((item) => {
      if (!item || !item.id) return;
      (item.builds_on || []).forEach((ref) => {
        if (ref && ref.id && byId[ref.id] && !referencedBy[ref.id].some((x) => x.item.id === item.id)) {
          referencedBy[ref.id].push({ item, note: ref.note || "" });
        }
      });
      if (item.parent_edition && byId[item.parent_edition] && !referencedBy[item.parent_edition].some((x) => x.item.id === item.id)) {
        referencedBy[item.parent_edition].push({
          item,
          note: `Companion publication for ${byId[item.parent_edition].short_title || item.parent_edition}.`,
        });
      }
    });

    const numbered = editions
      .filter((x) => x.is_edition && typeof x.issue_num === "number")
      .sort((a, b) => a.issue_num - b.issue_num);
    const latestEdition = numbered.length ? numbered[numbered.length - 1] : editions[0];

    if (navEl) {
      const preserved = [...navEl.querySelectorAll('a[data-keep-nav="1"]')].map((a) => a.outerHTML);
      const navEditions = editions.filter(
        (ed) => ed.is_edition || ed.id === "mali13-leaderboard" || ed.id === pageId
      );
      const navLinks = [
        `<a href="${esc(resolveSiteUrl("index.html"))}"${pageId === "front" ? ' class="active"' : ""}>Front Page</a>`,
        ...navEditions.map((ed) => {
          const isLatest = latestEdition && ed.id === latestEdition.id;
          const badge = isLatest ? `<span class="nav-badge">New</span>` : "";
          const label = esc(ed.short_title || ed.title);
          return `<a href="${esc(resolveSiteUrl(ed.url))}"${ed.id === pageId ? ' class="active"' : ""}>${label}${badge}</a>`;
        }),
        ...preserved,
      ];
      navEl.innerHTML = navLinks.join("");
    }

    document.querySelectorAll("a[data-ref-id]").forEach((a) => {
      const target = byId[a.dataset.refId];
      if (target) a.href = resolveSiteUrl(target.url);
    });

    const current = byId[pageId];
    if (!current || pageId === "front") return;

    const effectiveIssueNum =
      typeof current.issue_num === "number"
        ? current.issue_num
        : current.parent_edition && byId[current.parent_edition]
          ? byId[current.parent_edition].issue_num
          : null;

    if (!document.getElementById("chronicleLineageStrip")) {
      const anchorNav = document.querySelector("nav.ed-subnav") || navEl;
      if (anchorNav && anchorNav.parentNode) {
        const hasNewer =
          latestEdition &&
          typeof effectiveIssueNum === "number" &&
          typeof latestEdition.issue_num === "number" &&
          effectiveIssueNum < latestEdition.issue_num;
        const strip = document.createElement("div");
        strip.id = "chronicleLineageStrip";

        if (hasNewer) {
          strip.className = "lineage-strip";
          const followUps = [latestEdition, ...(referencedBy[current.id] || []).map((x) => x.item)].filter(
            (x, i, arr) => x && x.id !== current.id && arr.findIndex((y) => y.id === x.id) === i
          );
          strip.innerHTML = `
            <div>
              <span class="lineage-label">Newer Edition Published &rarr;</span>
              <span>This ${esc(current.vol_issue)} dispatch is followed by <strong>${esc(latestEdition.short_title)}</strong> (${esc(latestEdition.date)}).</span>
            </div>
            <div class="lineage-links">
              ${followUps
                .map(
                  (ed) =>
                    `<a class="lineage-pill" href="${esc(resolveSiteUrl(ed.url))}">${esc(ed.short_title || ed.title)} &rarr;</a>`
                )
                .join("")}
            </div>
          `;
          anchorNav.insertAdjacentElement("afterend", strip);
        } else {
          const backList = [...(current.builds_on || [])];
          if (current.parent_edition && byId[current.parent_edition] && !backList.some((x) => x.id === current.parent_edition)) {
            backList.unshift({ id: current.parent_edition, note: "" });
          }
          if (backList.length) {
            strip.className = "lineage-strip older-ref";
            strip.innerHTML = `
              <div>
                <span class="lineage-label">&larr; Builds On Earlier Dispatches</span>
                <span>Prior editions &amp; companions referenced by ${esc(current.short_title)}:</span>
              </div>
              <div class="lineage-links">
                ${backList
                  .map((b) => byId[b.id])
                  .filter(Boolean)
                  .map(
                    (ed) =>
                      `<a class="lineage-pill" href="${esc(resolveSiteUrl(ed.url))}">&larr; ${esc(ed.short_title || ed.title)}</a>`
                  )
                  .join("")}
              </div>
            `;
            anchorNav.insertAdjacentElement("afterend", strip);
          }
        }
      }
    }

    if (!document.getElementById("chronicleCrossNav")) {
      const sheet = document.querySelector(".sheet");
      if (!sheet) return;
      const colophon = sheet.querySelector("footer.colophon");

      const prevEd =
        typeof effectiveIssueNum === "number"
          ? [...numbered].reverse().find((x) => x.issue_num < effectiveIssueNum)
          : null;
      const nextEd =
        typeof effectiveIssueNum === "number"
          ? numbered.find((x) => x.issue_num > effectiveIssueNum)
          : null;

      const buildsList = [...(current.builds_on || [])];
      if (current.parent_edition && byId[current.parent_edition] && !buildsList.some((x) => x.id === current.parent_edition)) {
        buildsList.unshift({
          id: current.parent_edition,
          note: `Parent edition for ${current.short_title}.`,
        });
      }

      const forwardList = [...(referencedBy[current.id] || [])];
      numbered.forEach((ed) => {
        if (
          typeof effectiveIssueNum === "number" &&
          ed.issue_num > effectiveIssueNum &&
          ed.id !== current.id &&
          !forwardList.some((x) => x.item.id === ed.id)
        ) {
          forwardList.push({
            item: ed,
            note: `Successive issue published ${ed.date}.`,
          });
        }
      });

      const leftSlot = prevEd
        ? `<a class="pager-slot pk-link" href="${esc(resolveSiteUrl(prevEd.url))}"><span class="pk">&larr; Previous Edition</span><span>${esc(prevEd.vol_issue)} &bull; ${esc(prevEd.short_title)}</span></a>`
        : `<div class="pager-slot muted"><span class="pk">&larr; Series Start</span><span>Inaugural Vol. I, No. 1</span></div>`;

      const rightSlot = nextEd
        ? `<a class="pager-slot right pk-link" href="${esc(resolveSiteUrl(nextEd.url))}"><span class="pk">Newer Edition &rarr;</span><span>${esc(nextEd.vol_issue)} &bull; ${esc(nextEd.short_title)}</span></a>`
        : `<div class="pager-slot right muted"><span class="pk">Current Issue</span><span>Latest Published Edition</span></div>`;

      const centerSlot = `<a class="pager-slot center pk-link" href="${esc(resolveSiteUrl("index.html#archive"))}"><span class="pk">Chronicle Index</span><span>Front Page &amp; Archive</span></a>`;

      const colsHtml = [];
      if (buildsList.length) {
        colsHtml.push(`
          <div class="cont-col">
            <h4>Builds On &amp; Earlier Foundations</h4>
            <ul>
              ${buildsList
                .map((b) => {
                  const target = byId[b.id];
                  if (!target) return "";
                  return `<li><a href="${esc(resolveSiteUrl(target.url))}"><strong>${esc(target.short_title || target.title)}</strong></a> <code>${esc(target.vol_issue)}</code>${b.note ? ` : ${esc(b.note)}` : ""}</li>`;
                })
                .join("")}
            </ul>
          </div>
        `);
      }
      if (forwardList.length) {
        colsHtml.push(`
          <div class="cont-col">
            <h4>Continued In &amp; Referenced By</h4>
            <ul>
              ${forwardList
                .map(
                  ({ item, note }) =>
                    `<li><a href="${esc(resolveSiteUrl(item.url))}"><strong>${esc(item.short_title || item.title)}</strong></a> <code>${esc(item.vol_issue)}</code>${note ? ` : ${esc(note)}` : ""}</li>`
                )
                .join("")}
            </ul>
          </div>
        `);
      }

      const sec = document.createElement("section");
      sec.id = "chronicleCrossNav";
      sec.className = "chronicle-continuity";
      sec.setAttribute("aria-label", "Chronicle Edition Continuity");
      sec.innerHTML = `
        <div class="cont-head">
          <h3>Chronicle Continuity &amp; Cross-Edition Index</h3>
          <span class="tag ${esc(current.tagCls || "sebeni")}">${esc(current.vol_issue)}</span>
        </div>
        <div class="continuity-pager">${leftSlot}${centerSlot}${rightSlot}</div>
        ${colsHtml.length ? `<div class="continuity-grid${colsHtml.length === 1 ? " single" : ""}">${colsHtml.join("")}</div>` : ""}
        <div class="continuity-pills">
          <span style="font-weight:700;color:var(--ink-soft);margin-right:4px">All Dispatches:</span>
          <a href="${esc(resolveSiteUrl("index.html"))}">Front Page</a>
          ${editions
            .map(
              (ed) =>
                `<a href="${esc(resolveSiteUrl(ed.url))}"${ed.id === current.id ? ' class="active"' : ""}>${esc(ed.short_title || ed.title)}</a>`
            )
            .join("")}
        </div>
      `;
      if (colophon) colophon.parentNode.insertBefore(sec, colophon);
      else sheet.appendChild(sec);
    }
  }

  window.Chronicle = { esc, fetchJson, resolveSiteUrl };
  document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    hydrateCrossEditionNav();
    hydrateFrontPage();
  });
})();
