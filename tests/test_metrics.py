"""Unit tests for MALI13 scoring formulas, strict DabaX guarding, and Pareto extraction."""

from __future__ import annotations

from bench import LANGS
from bench.metrics import Mali13Scorer, pareto_frontier, strip_thinking
from bench.schema import (
    ClozeSample,
    CompletionRecord,
    EfficiencyStats,
    GenSample,
    LidSample,
    ModelScorecard,
    ModelSpec,
    MorphSample,
    RefToken,
)


def test_strip_thinking_removes_think_tags():
    raw = '<think>internal reasoning about Bambara\nmore text</think>{"lang": "bam"}'
    assert strip_thinking(raw) == '{"lang": "bam"}'
    unclosed = '<think>cut off mid thought'
    assert strip_thinking(unclosed) == ""


def test_lid_f1_and_confusion_matrix():
    scorer = Mali13Scorer()
    samples = [LidSample(id=f"{l}-lid-1", lang=l, text="x") for l in LANGS]
    recs = {
        s.id: CompletionRecord(
            sample_id=s.id,
            task="lid",
            lang=s.lang,
            completion=f'{{"lang": "{s.lang}"}}',
        )
        for s in samples
    }
    f1s, mat = scorer.score_lid(samples, recs)
    assert all(v == 100.0 for v in f1s.values())
    assert sum(mat[i][i] for i in range(13)) == 13


def test_cloze_extract_json_and_logprob():
    scorer = Mali13Scorer()
    samples = [
        ClozeSample(
            id="bam-cloze-1",
            lang="bam",
            masked_text="A [MASK] taa.",
            target="bɛ",
            choices={"A": "ka", "B": "bɛ", "C": "tɛ", "D": "na"},
            answer="B",
        ),
        ClozeSample(
            id="ful-cloze-1",
            lang="ful",
            masked_text="Yimɓe [MASK].",
            target="suɓaaɓe",
            choices={"A": "suɓaa", "B": "suɓaaɓe", "C": "suɓoowo", "D": "suɓi"},
            answer="B",
        ),
    ]
    recs = {
        "bam-cloze-1": CompletionRecord(
            sample_id="bam-cloze-1", task="cloze", lang="bam", completion='{"choice": "B"}'
        ),
        "ful-cloze-1": CompletionRecord(
            sample_id="ful-cloze-1",
            task="cloze",
            lang="ful",
            cloze_mode="logprob",
            logprob_scores={"A": -3.2, "B": -0.4, "C": -4.1, "D": -5.0},
        ),
    }
    scores = scorer.score_cloze(samples, recs)
    assert scores["bam"] == 100.0
    assert scores["ful"] == 100.0


def test_morph_rejects_hallucinated_stages_on_garbage():
    scorer = Mali13Scorer()
    sample = MorphSample(
        id="bam-morph-1",
        lang="bam",
        text="A ka taa",
        ref_tokens=[
            RefToken(surface="A", stage="0", morphemes=["a"]),
            RefToken(surface="ka", stage="0", morphemes=["ka"]),
            RefToken(surface="taa", stage="0", morphemes=["taa"]),
        ],
        ref_morphemes=["a", "ka", "taa"],
        ref_stages=["0", "0", "0"],
        ref_strict_rate=1.0,
        n_inflected=0,
    )
    # Model claims stage 0 on completely unparseable nonsense token
    bad_rec = CompletionRecord(
        sample_id="bam-morph-1",
        task="morph",
        lang="bam",
        completion='{"tokens": [{"surface": "babalakafata", "stage": "0", "morphemes": ["babalakafata"]}]}',
    )
    scores = scorer.score_morph([sample], {"bam-morph-1": bad_rec})
    # StrictRate and Phi must be 0.0 because DabaX(strict_parse=True) recomputes stage = -1
    assert scores["bam"] == 0.0


def test_scorecard_equity_and_pareto_frontier():
    scorer = Mali13Scorer()
    spec = ModelSpec(
        slug="m1",
        display_name="Model 1",
        provider="mock",
        model_id="mock",
        track="Sebeni",
        active_params_b=1.0,
    )
    eff = EfficiencyStats(
        word_fertility=2.0,
        morpheme_fertility=1.5,
        bytes_per_token=3.2,
        diacritic_penalty=1.4,
        ascii_fertility=1.7,
        diacritic_fertility=2.38,
        bpb=1.1,
        per_lang_fertility={l: 2.0 for l in LANGS},
    )
    m_samples = [
        MorphSample(
            id=f"{l}-m-1", lang=l, text="A", ref_tokens=[], ref_morphemes=["a"],
            ref_stages=["0"], ref_strict_rate=1.0, n_inflected=0
        )
        for l in LANGS
    ]
    g_samples = [GenSample(id=f"{l}-g-1", lang=l, prompt_prefix="A", ref_continuation="B") for l in LANGS]
    l_samples = [LidSample(id=f"{l}-l-1", lang=l, text="A") for l in LANGS]
    c_samples = [
        ClozeSample(id=f"{l}-c-1", lang=l, masked_text="[MASK]", target="x", choices={"A":"x","B":"y","C":"z","D":"w"}, answer="A")
        for l in LANGS
    ]
    card, _, _ = scorer.build_scorecard(spec, m_samples, g_samples, l_samples, c_samples, [], eff)
    assert card.q == 0.0
    assert card.q_min3 == 0.0

    c1 = ModelScorecard(
        slug="a", id="A", provider="mock", model_id="a", track="Sebeni", trackCls="sebeni",
        q=75.0, q_min3=60.0, q_gap=20.0, morph=75, gen=75, lid=75, cloze=75,
        fert=2.1, morph_fert=1.6, bytes_per_tok=3.0, diac=1.4, ascii_fert=1.8, ext_fert=2.5,
        lat=0.5, tok_s=40.0, cost=0.0,
    )
    c2 = c1.model_copy(update={"slug": "b", "id": "B", "q": 70.0, "fert": 2.4})  # dominated by A
    c3 = c1.model_copy(update={"slug": "c", "id": "C", "q": 80.0, "fert": 2.5})  # higher Q, higher fert -> Pareto
    assert pareto_frontier([c1, c2, c3], "fert") == ["a", "c"]
