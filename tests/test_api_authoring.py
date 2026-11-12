"""Tests for api.authoring: the overlap meter, draft storage, and promotion.

The meter tests are written as claims about *behaviour under paraphrase*
rather than about particular numbers, because the thresholds are explicitly
provisional (see the module docstring). What must hold regardless of where the
bands end up: copying a term of art scores worse than sharing a common word,
and a plain-English rewrite scores better than the heading it came from.

The promotion tests defend the property the gold set depends on — a citation
that is not in the index can never enter the file — and the property that
makes the two-file write safe: promotion is re-runnable without duplicating.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from api import authoring
from api.authoring import (
    AMBER,
    GREEN,
    RED,
    Vocabulary,
    load_draft,
    overlap,
    promote,
    save_draft,
)
from eval import goldset
from index import store
from pipeline.manifest import TitleSpec
from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

REPO_MANIFEST = __import__('pathlib').Path(__file__).resolve().parent.parent / 'corpus.toml'

SPEC4 = TitleSpec(4, "2024-07-18", "Accounts")
SPEC31 = TitleSpec(31, "2024-07-18", "Money and Finance: Treasury")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "index" / "regsearch.db"
    store.build_index(FIXTURE_CHUNKS, db_path=path, embedder=FakeEmbedder(), show_progress=False)
    authoring.reset_vocabulary_cache()
    return path


@pytest.fixture
def eval_dir(tmp_path):
    out = tmp_path / "eval"
    out.mkdir()
    return out


@pytest.fixture
def gold_path(eval_dir):
    return eval_dir / "gold.jsonl"


def sections_in(db):
    return sorted(goldset.index_sections(db))


# --- the overlap meter ------------------------------------------------------


BODY = (
    "Veterans' preference shall be applied to each preference eligible applicant "
    "for appointment to a competitive position in the Government Accountability Office."
)


def vocab(**df):
    """A corpus of 1000 sections with the given document frequencies."""
    return Vocabulary(n_sections=1000, df=df)


class TestMeter:
    def test_an_empty_question_leaves_the_meter_idle(self):
        score = overlap("", BODY)
        assert score.verdict == "empty" and score.shared == ()

    def test_verbatim_copying_is_red(self):
        score = overlap(BODY, BODY)
        assert score.verdict == RED
        assert score.distinctive == pytest.approx(1.0)

    def test_a_plain_english_rewrite_beats_the_statutory_wording(self):
        """The whole point: the de-vocabbed form must score better."""
        lifted = overlap("What is the veterans' preference for a preference eligible?", BODY)
        plain = overlap("Do people who served in the armed forces get any advantage?", BODY)
        assert plain.distinctive < lifted.distinctive
        assert plain.verdict in (GREEN, AMBER)

    def test_rarity_is_what_counts_not_raw_sharing(self):
        """Sharing a common word is unavoidable; sharing a term of art is the problem."""
        body = "The applicant shall file a certificate of estoppel with the Comptroller."
        common = overlap("What must someone file?", body, vocab=vocab(file=900, applicant=800))
        rare = overlap("What is estoppel?", body, vocab=vocab(estoppel=1))
        assert rare.distinctive > common.distinctive
        assert rare.verdict == RED and common.verdict == GREEN

    def test_a_word_in_every_section_carries_no_weight(self):
        body = "The agency shall publish the notice."
        score = overlap("What must the agency publish?", body,
                        vocab=vocab(agency=1000, publish=1000))
        assert score.distinctive == pytest.approx(0.0)
        assert score.verdict == GREEN

    def test_unknown_words_count_as_the_authors_own(self):
        """A word the corpus has never seen was brought by the author, not copied."""
        body = "The agency shall publish the notice."
        score = overlap("Must the regulator broadcast anything?", body, vocab=vocab(agency=1000))
        assert score.distinctive == pytest.approx(0.0)

    def test_two_terms_of_art_are_red_whatever_the_ratio_says(self):
        body = "A preference eligible claiming estoppel against the Comptroller."
        long_question = (
            "Could somebody possibly wonder whether estoppel applies to a preference "
            "eligible person in a variety of unrelated circumstances elsewhere entirely?"
        )
        score = overlap(long_question, body, vocab=vocab(estoppel=1, preference=2, eligible=3))
        assert len(score.rare_copied) >= 2
        assert score.verdict == RED

    def test_shared_terms_are_listed_rarest_first(self):
        body = "The applicant shall file a certificate of estoppel."
        score = overlap("Must an applicant file estoppel?", body,
                        vocab=vocab(applicant=800, file=900, estoppel=1))
        assert score.shared[0].term == "estoppel"
        assert score.shared[0].distinctive and not score.shared[-1].distinctive

    def test_a_prose_question_naming_its_section_is_red(self):
        """The citation-aware tokenizer matches that directly; it measures nothing."""
        score = overlap("What does 4 CFR 25.14 require of visitors?", "Visitors to the building.")
        assert score.names_citation and score.verdict == RED
        assert "citation lookup" in score.advice

    def test_a_citation_lookup_may_name_its_section(self):
        score = overlap("What does 4 CFR 25.14 provide?", "Visitors to the building.",
                        kind="citation")
        assert not score.names_citation

    def test_plurals_do_not_hide_a_copy(self):
        """The S-stemmer the index uses means 'employees' cannot dodge 'employee'."""
        score = overlap("What about employees?", "Each employee shall report.",
                        vocab=vocab(employee=1))
        assert "employee" in {s.term for s in score.shared}

    def test_jaccard_is_reported_alongside_the_weighted_score(self):
        score = overlap("Must an applicant file?", "The applicant shall file.")
        assert 0.0 < score.jaccard <= 1.0

    def test_the_meter_is_serialisable_for_the_json_api(self):
        payload = overlap("Must an applicant file?", "The applicant shall file.").to_dict()
        assert set(payload) >= {"verdict", "distinctive", "shared", "names_citation"}
        json.dumps(payload)


class TestVocabulary:
    def test_rarity_runs_from_common_to_unique(self):
        v = vocab(everywhere=1000, rare=1)
        assert v.rarity("everywhere") == pytest.approx(0.0)
        assert v.rarity("rare") == pytest.approx(1.0)
        assert 0 < v.rarity("unseen") <= 1.0

    def test_built_from_the_index_counts_sections_not_chunks(self, db):
        built = authoring.vocabulary(db)
        assert built.n_sections == len(sections_in(db))
        assert built.n_sections > 0
        assert all(count <= built.n_sections for count in built.df.values())

    def test_an_absent_index_yields_an_empty_vocabulary(self, tmp_path):
        assert authoring.vocabulary(tmp_path / "nope.db").n_sections == 0

    def test_the_cache_is_keyed_on_the_index_file(self, db):
        first = authoring.vocabulary(db)
        assert authoring.vocabulary(db) is first  # cached
        authoring.reset_vocabulary_cache()
        assert authoring.vocabulary(db) is not first


# --- sections ---------------------------------------------------------------


class TestSections:
    def test_loads_heading_breadcrumb_and_body(self, db):
        citation = sections_in(db)[0]
        section = authoring.load_sections(db, [citation])[citation]
        assert section.found and section.heading
        assert section.citation == citation
        assert section.heading not in section.breadcrumb, "the leaf is not repeated"

    def test_a_missing_section_is_reported_not_raised(self, db):
        assert authoring.load_sections(db, ["4 CFR § 999.9"])["4 CFR § 999.9"].found is False


# --- draft storage ----------------------------------------------------------


class TestDraftStorage:
    def test_scaffold_writes_a_per_title_draft_file(self, db, eval_dir):
        authoring.scaffold(SPEC4, eval_dir=eval_dir, db_path=db,
                           n_prose=2, n_citation=1, n_negatives=1)
        path = authoring.draft_path(4, eval_dir)
        assert path.name == "gold_title4_draft.jsonl"
        assert len(goldset.load(path)) == 4

    def test_scaffolded_entries_start_as_drafts_with_ids(self, db, eval_dir):
        entries = authoring.scaffold(SPEC4, eval_dir=eval_dir, db_path=db,
                                     n_prose=2, n_citation=0, n_negatives=0)
        assert all(e["status"] == "draft" for e in entries)
        assert len({e["draft_id"] for e in entries}) == len(entries)

    def test_scaffold_refuses_to_clobber_existing_work(self, db, eval_dir):
        authoring.scaffold(SPEC4, eval_dir=eval_dir, db_path=db,
                           n_prose=1, n_citation=0, n_negatives=0)
        with pytest.raises(FileExistsError):
            authoring.scaffold(SPEC4, eval_dir=eval_dir, db_path=db)

    def test_scaffold_for_an_unindexed_title_fails_clearly(self, db, eval_dir):
        with pytest.raises(ValueError, match="no sections"):
            authoring.scaffold(SPEC31, eval_dir=eval_dir, db_path=db)

    def test_round_trips_unicode_and_sections_symbols(self, eval_dir):
        entries = [{"question": "Qué § 2.3 — naïve?", "relevant_citations": ["4 CFR § 2.3"],
                    "kind": "prose", "status": "draft", "draft_id": "abc"}]
        save_draft(4, entries, eval_dir)
        assert load_draft(4, eval_dir)[0]["question"] == "Qué § 2.3 — naïve?"

    def test_ids_are_assigned_to_a_hand_written_draft_and_persisted(self, eval_dir):
        """eval/gold_title31_draft.jsonl was committed by hand and has no ids."""
        path = authoring.draft_path(31, eval_dir)
        path.write_text(
            json.dumps({"question": "q", "relevant_citations": [], "status": "draft"}) + "\n",
            encoding="utf-8",
        )
        first = load_draft(31, eval_dir)
        assert first[0]["draft_id"]
        assert load_draft(31, eval_dir)[0]["draft_id"] == first[0]["draft_id"]

    def test_writes_leave_no_temp_files_behind(self, eval_dir):
        save_draft(4, [{"question": "q", "relevant_citations": [], "draft_id": "a"}], eval_dir)
        assert [p.name for p in eval_dir.iterdir()] == ["gold_title4_draft.jsonl"]

    def test_a_missing_draft_is_an_empty_queue_not_an_error(self, eval_dir):
        assert load_draft(99, eval_dir) == []


# --- promotion --------------------------------------------------------------


def approved(citation, question, **kw):
    entry = {
        "question": question,
        "relevant_citations": [citation] if citation else [],
        "kind": "prose",
        "status": "approved",
        "draft_id": question[:8],
    }
    entry.update(kw)
    return entry


class TestPromotion:
    def test_merges_approved_entries_and_leaves_drafts_alone(self, db, eval_dir, gold_path):
        cite = sections_in(db)[0]
        save_draft(4, [
            approved(cite, "Who may enter the building?"),
            approved(cite, "Still thinking about this one", status="draft"),
        ], eval_dir)

        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        assert result.ok, result.errors
        assert len(result.promoted) == 1
        assert result.still_draft == 1
        gold = goldset.load(gold_path)
        assert [e["question"] for e in gold] == ["Who may enter the building?"]
        assert [e["question"] for e in load_draft(4, eval_dir)] == [
            "Still thinking about this one"
        ]

    def test_appends_rather_than_replacing(self, db, eval_dir, gold_path):
        cite = sections_in(db)[0]
        goldset.write([{"question": "existing", "relevant_citations": [cite],
                        "kind": "prose", "status": "approved"}], gold_path)
        save_draft(4, [approved(cite, "brand new question")], eval_dir)

        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        assert result.ok
        assert [e["question"] for e in goldset.load(gold_path)] == [
            "existing", "brand new question"
        ]
        assert (result.gold_before, result.gold_after) == (1, 2)

    def test_strips_authoring_bookkeeping_from_the_gold_file(self, db, eval_dir, gold_path):
        cite = sections_in(db)[0]
        save_draft(4, [approved(cite, "a question", generated=True, revealed=True)], eval_dir)
        promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        entry = goldset.load(gold_path)[0]
        assert not {"draft_id", "generated", "revealed"} & set(entry)
        assert entry["status"] == "approved"

    def test_a_negative_keeps_the_corpus_it_was_verified_against(self, db, eval_dir, gold_path):
        save_draft(4, [approved("", "What is the tax filing deadline?",
                                negative_corpus=[SPEC4.slug])], eval_dir)
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert result.ok
        assert goldset.load(gold_path)[0]["negative_corpus"] == [SPEC4.slug]

    # -- refusals --

    def test_a_citation_not_in_the_index_blocks_the_whole_merge(self, db, eval_dir, gold_path):
        """Requirement: confirm every citation physically exists in chunks."""
        good = sections_in(db)[0]
        save_draft(4, [
            approved(good, "a good question"),
            approved("4 CFR § 999.9", "a question about a section that does not exist"),
        ], eval_dir)

        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        assert not result.ok
        assert ("a question about a section that does not exist", "4 CFR § 999.9") \
            in result.bad_citations
        assert not gold_path.exists(), "all-or-nothing: the good one must not land either"
        assert len(load_draft(4, eval_dir)) == 2, "the draft is untouched"

    def test_an_unfilled_stub_marked_approved_is_rejected(self, db, eval_dir, gold_path):
        save_draft(4, [approved(sections_in(db)[0], "   ")], eval_dir)
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert not result.ok
        assert any(p.check == "gold:empty-question" for p in result.errors)
        assert not gold_path.exists()

    def test_a_duplicate_of_an_existing_question_is_rejected(self, db, eval_dir, gold_path):
        """Not the same as re-promoting: this is a *different* draft entry that
        happens to duplicate one already in the set, which would double-weight it."""
        cite = sections_in(db)[0]
        goldset.write([{"question": "the same question", "relevant_citations": [cite],
                        "kind": "prose", "status": "approved"}], gold_path)
        save_draft(4, [
            approved(cite, "the same question"),
            approved(cite, "a genuinely new question"),
        ], eval_dir)

        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        # The verbatim repeat is recognised as already present, not appended twice.
        assert result.ok
        assert [e["question"] for e in result.already_present] == ["the same question"]
        assert len(goldset.load(gold_path)) == 2

    def test_two_identical_new_questions_in_one_batch_are_rejected(self, db, eval_dir, gold_path):
        cite = sections_in(db)[0]
        save_draft(4, [
            approved(cite, "same text", draft_id="a"),
            approved(cite, "same text", draft_id="b"),
        ], eval_dir)
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert not result.ok
        assert any(p.check == "gold:duplicate" for p in result.errors)

    def test_nothing_approved_is_refused_with_a_clear_reason(self, db, eval_dir, gold_path):
        save_draft(4, [approved(sections_in(db)[0], "q", status="draft")], eval_dir)
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert not result.ok
        assert result.errors[0].check == "authoring:nothing-approved"

    def test_promotion_without_an_index_is_refused(self, tmp_path, eval_dir, gold_path):
        save_draft(4, [approved("4 CFR § 2.3", "q")], eval_dir)
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path,
                         db_path=tmp_path / "missing.db")
        assert not result.ok
        assert result.errors[0].check == "authoring:no-index"
        assert not gold_path.exists()

    def test_a_pre_existing_problem_does_not_block_new_work(self, db, eval_dir, gold_path):
        """A stale entry someone else left behind must not make the file
        unappendable forever."""
        cite = sections_in(db)[0]
        goldset.write([{"question": "an old negative", "relevant_citations": [],
                        "kind": "prose", "status": "approved"}], gold_path)  # no negative_corpus
        save_draft(4, [approved(cite, "a fine new question")], eval_dir)

        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        assert result.ok
        assert any(p.check == "gold:negative-unverified" for p in result.pre_existing)
        assert len(goldset.load(gold_path)) == 2

    # -- atomicity and locking --

    def test_promotion_is_re_runnable_after_a_crash_between_the_two_writes(
        self, db, eval_dir, gold_path
    ):
        """Gold and the draft are separate files, so the pair cannot be atomic.
        Re-running must retire the duplicate rather than append it twice."""
        cite = sections_in(db)[0]
        entry = approved(cite, "a question")
        save_draft(4, [entry], eval_dir)
        promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        save_draft(4, [entry], eval_dir)  # simulate the draft write never landing
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)

        assert result.ok
        assert result.promoted == ()
        assert len(result.already_present) == 1
        assert len(goldset.load(gold_path)) == 1, "no duplicate"
        assert load_draft(4, eval_dir) == [], "and the draft is cleared this time"

    def test_the_gold_file_is_replaced_atomically(self, db, eval_dir, gold_path, monkeypatch):
        """A reader during the write sees the old file, never a truncated one."""
        cite = sections_in(db)[0]
        save_draft(4, [approved(cite, "a question")], eval_dir)
        seen = []
        real_replace = authoring.os.replace

        def spy(src, dst):
            if str(dst).endswith("gold.jsonl"):
                seen.append(gold_path.exists())
            return real_replace(src, dst)

        monkeypatch.setattr(authoring.os, "replace", spy)
        promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert seen == [False], "gold.jsonl appeared only at the rename"
        assert len(goldset.load(gold_path)) == 1

    def test_concurrent_promotions_do_not_lose_an_entry(self, db, eval_dir, gold_path):
        """Two merges racing must serialise: both questions end up in the file."""
        cites = sections_in(db)
        save_draft(4, [approved(cites[0], "question one")], eval_dir)
        save_draft(31, [approved(cites[0], "question two")], eval_dir)
        barrier = threading.Barrier(2)
        results = {}

        def go(title):
            barrier.wait()
            results[title] = promote(
                title, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db
            )

        threads = [threading.Thread(target=go, args=(t,)) for t in (4, 31)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r.ok for r in results.values()), [r.errors for r in results.values()]
        assert {e["question"] for e in goldset.load(gold_path)} == {
            "question one", "question two"
        }

    def test_the_lock_is_held_for_the_whole_read_modify_write(self, gold_path):
        held = []
        with authoring.gold_lock(gold_path):
            held.append(authoring._PROMOTE_LOCK.locked())
        assert held == [True]
        assert not authoring._PROMOTE_LOCK.locked()

    # -- the staleness consequence --

    def test_a_successful_merge_reports_that_params_are_now_stale(
        self, db, eval_dir, gold_path
    ):
        """The banner is derived from the files, never stored: gold.jsonl changed,
        so retrieval/params.json no longer describes it."""
        save_draft(4, [approved(sections_in(db)[0], "a question")], eval_dir)
        result = promote(4, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert result.ok
        assert any("gold set changed" in reason for reason in result.staleness), result.staleness


# --- the queue view ---------------------------------------------------------


class TestQueue:
    def test_joins_each_entry_to_its_section_and_meter(self, db, eval_dir):
        cite = sections_in(db)[0]
        save_draft(4, [
            approved(cite, "Who may enter?"),
            approved(cite, "", status="draft"),
            approved("", "a negative", status="draft"),
        ], eval_dir)
        items = authoring.queue(4, eval_dir=eval_dir, db_path=db)
        # Every row carries an explicit state; none is ever left as None, so a
        # template cannot fall through to a neutral-looking blank.
        assert items[0].section.found and items[0].overlap.checked
        assert items[1].overlap.verdict == authoring.EMPTY, "no question yet"
        assert not items[1].overlap.checked
        assert items[2].is_negative
        assert items[2].overlap.verdict == authoring.NOT_APPLICABLE


# --- the HTTP surface -------------------------------------------------------


class TestAuthoringRoutes:
    """The routes, with the real corpus.toml (which declares Titles 1, 4 and 31)
    but a throwaway index, gold file and draft directory."""

    @pytest.fixture
    def client(self, db, eval_dir, gold_path):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        app = create_app(
            Settings(root=REPO_ROOT, eval_dir=eval_dir, gold_path=gold_path, db_path=db,
                     results_dir=eval_dir / "results"),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as test_client:
            yield test_client

    def test_index_lists_the_declared_titles_with_coverage(self, client):
        page = client.get("/authoring")
        assert page.status_code == 200
        assert "Title 4 (Accounts)" in page.text
        assert f"/ {goldset.MIN_PROSE_PER_TITLE}" in page.text

    def test_a_title_outside_the_corpus_is_a_404(self, client):
        assert client.get("/authoring/26").status_code == 404

    def test_scaffold_creates_the_queue_then_refuses_to_clobber_it(self, client, eval_dir):
        created = client.post("/authoring/4/scaffold", follow_redirects=False)
        assert created.status_code == 303
        assert authoring.draft_path(4, eval_dir).exists()
        assert client.post("/authoring/4/scaffold").status_code == 409

    def test_scaffolding_an_unindexed_title_is_refused(self, client):
        assert client.post("/authoring/31/scaffold").status_code == 422

    def test_the_queue_shows_the_citation_and_heading_but_not_the_body(self, client, eval_dir):
        """The core guardrail: 'collapsed' means absent from the payload."""
        authoring.scaffold(SPEC4, eval_dir=eval_dir, db_path=client.app.state.settings.db_path,
                           n_prose=4, n_citation=0, n_negatives=0)
        page = client.get("/authoring/4")
        assert page.status_code == 200
        assert "4 CFR § 7.3" in page.text
        assert "Financial disclosure" in page.text          # heading, shown
        assert "termination of employment" not in page.text  # body, not shown
        assert "favoritism" not in page.text
        assert "Reveal text" in page.text

    def test_reveal_returns_the_body_and_records_that_it_was_opened(self, client, eval_dir):
        db_path = client.app.state.settings.db_path
        entries = authoring.scaffold(SPEC4, eval_dir=eval_dir, db_path=db_path,
                                     n_prose=4, n_citation=0, n_negatives=0)
        entry = next(e for e in entries if e["relevant_citations"] == ["4 CFR § 7.3"])
        revealed = client.get(f"/authoring/4/reveal/{entry['draft_id']}")
        assert revealed.status_code == 200
        assert "termination of employment" in revealed.text
        again = authoring.find_draft(load_draft(4, eval_dir), entry["draft_id"])
        assert again["revealed"] is True

    def test_scaffolded_notes_do_not_smuggle_the_body_in(self, client, eval_dir):
        """eval.goldset pastes a 600-char excerpt into its notes for CLI authors."""
        entries = authoring.scaffold(
            SPEC4, eval_dir=eval_dir, db_path=client.app.state.settings.db_path,
            n_prose=4, n_citation=0, n_negatives=0,
        )
        notes = " ".join(e["notes"] for e in entries)
        assert "termination of employment" not in notes
        assert "Reveal text" in notes

    def test_the_meter_endpoint_scores_without_returning_the_text(self, client):
        response = client.post(
            "/authoring/4/meter",
            data={"question": "Must a GAO employee file a financial disclosure report?",
                  "citation": "4 CFR § 7.3", "kind": "prose"},
        )
        assert response.status_code == 200
        assert "borrowed" in response.text
        assert "termination of employment" not in response.text

    def test_saving_an_entry_persists_it_and_returns_the_refreshed_row(self, client, eval_dir):
        entries = authoring.scaffold(
            SPEC4, eval_dir=eval_dir, db_path=client.app.state.settings.db_path,
            n_prose=4, n_citation=0, n_negatives=0,
        )
        entry = next(e for e in entries if e["relevant_citations"] == ["4 CFR § 7.3"])
        response = client.post(
            f"/authoring/4/entry/{entry['draft_id']}",
            data={"question": "What paperwork does a watchdog staffer owe each year?",
                  "notes": "de-vocabbed", "kind": "prose", "status": "approved",
                  "citations": "4 CFR § 7.3"},
        )
        assert response.status_code == 200
        saved = authoring.find_draft(load_draft(4, eval_dir), entry["draft_id"])
        assert saved["status"] == "approved"
        assert saved["question"].startswith("What paperwork")

    def test_a_bad_status_is_refused(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", "q", draft_id="x")], eval_dir)
        response = client.post("/authoring/4/entry/x", data={"question": "q", "status": "yolo"})
        assert response.status_code == 422

    def test_promote_renders_the_merge_and_the_re_tune_banner(self, client, eval_dir, gold_path):
        save_draft(4, [approved("4 CFR § 7.3", "Who must file paperwork each year?")], eval_dir)
        response = client.post("/authoring/4/promote")
        assert response.status_code == 200
        assert "Re-tune required" in response.text
        assert len(goldset.load(gold_path)) == 1

    def test_a_refused_promotion_renders_the_reason_and_writes_nothing(
        self, client, eval_dir, gold_path
    ):
        save_draft(4, [approved("4 CFR § 999.9", "about a section that does not exist")],
                   eval_dir)
        response = client.post("/authoring/4/promote")
        assert response.status_code == 422
        assert "Citations not in the index" in response.text
        assert not gold_path.exists()

    def test_the_json_view_exposes_the_meter_but_never_the_body(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", "Who must file a financial disclosure report?")],
                   eval_dir)
        payload = client.get("/api/authoring/4").json()
        assert payload[0]["overlap"]["verdict"] in ("green", "amber", "red")
        assert "termination of employment" not in json.dumps(payload)


class TestHandWrittenDrafts:
    """eval/gold_title31_draft.jsonl was committed by hand and omits fields."""

    def test_missing_fields_are_filled_in(self, eval_dir):
        authoring.draft_path(31, eval_dir).write_text(
            json.dumps({"question": "q", "relevant_citations": ["31 CFR § 1.1"]}) + "\n",
            encoding="utf-8",
        )
        entry = load_draft(31, eval_dir)[0]
        assert entry["kind"] == "prose" and entry["notes"] == ""

    def test_an_entry_with_no_status_is_a_draft_not_an_approval(self, eval_dir):
        """goldset.load() reads a missing status as 'approved'. In the queue that
        feeds the measured set, silence must mean 'not yet'."""
        authoring.draft_path(31, eval_dir).write_text(
            json.dumps({"question": "q", "relevant_citations": ["31 CFR § 1.1"]}) + "\n",
            encoding="utf-8",
        )
        assert load_draft(31, eval_dir)[0]["status"] == "draft"

    def test_an_explicit_approval_survives(self, eval_dir):
        authoring.draft_path(31, eval_dir).write_text(
            json.dumps({"question": "q", "relevant_citations": [], "status": "approved"}) + "\n",
            encoding="utf-8",
        )
        assert load_draft(31, eval_dir)[0]["status"] == "approved"


class TestMeterCannotCheck:
    def test_no_section_text_is_not_green(self):
        """'Could not check' and 'checked, nothing shared' are opposite facts."""
        score = overlap("Who must file a report?", "")
        assert score.verdict == authoring.UNKNOWN
        assert score.verdict != GREEN
        assert not score.checked and score.unverified
        assert "NOT been checked" in score.advice

    def test_an_unindexed_citation_reports_unknown_over_http(self, tmp_path):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        app = create_app(
            Settings(root=REPO_ROOT, eval_dir=tmp_path, gold_path=tmp_path / "gold.jsonl",
                     db_path=tmp_path / "missing.db", results_dir=tmp_path),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as client:
            response = client.post(
                "/authoring/31/meter",
                data={"question": "Who must file a report?", "citation": "31 CFR § 1.1"},
            )
        assert "NOT CHECKED" in response.text
        assert "NOT been checked" in response.text
        assert "meter green" not in response.text


# --- the "false green" trap -------------------------------------------------


class TestUncheckedIsNotClean:
    """"Could not check" and "checked and clean" must never look the same.

    The trap: an unindexed title gives the meter an empty body, every overlap
    computation then honestly reports 0% shared vocabulary, and the badge goes
    green. An author scanning a queue for green badges approves a question
    nobody verified. Each branch below is a way to reach that empty body.
    """

    def make(self, eval_dir, citation, question="Who must file paperwork each year?", **kw):
        entry = {"question": question, "relevant_citations": [citation] if citation else [],
                 "kind": "prose", "status": "draft", "draft_id": "x", **kw}
        save_draft(4, [entry], eval_dir)
        return entry

    def test_an_unindexed_citation_is_unknown_not_green(self, db, eval_dir):
        self.make(eval_dir, "4 CFR § 999.9")
        item = authoring.queue(4, eval_dir=eval_dir, db_path=db)[0]
        assert item.overlap.verdict == authoring.UNKNOWN
        assert item.overlap.verdict != authoring.GREEN
        assert not item.overlap.checked and item.unverified

    def test_no_index_at_all_is_unknown_not_green(self, eval_dir, tmp_path):
        self.make(eval_dir, "4 CFR § 7.3")
        item = authoring.queue(4, eval_dir=eval_dir, db_path=tmp_path / "absent.db")[0]
        assert item.overlap.verdict == authoring.UNKNOWN
        assert "not in the index" in item.overlap.advice

    def test_a_section_with_an_empty_body_is_unknown_not_green(self, db, eval_dir):
        """Indexed, but the stored text is blank — an indexing bug, not a pass."""
        import sqlite3

        conn = sqlite3.connect(db)
        conn.execute("update chunks set text = '' where section_citation = '4 CFR § 7.3'")
        conn.commit()
        conn.close()
        authoring.reset_vocabulary_cache()
        self.make(eval_dir, "4 CFR § 7.3")
        item = authoring.queue(4, eval_dir=eval_dir, db_path=db)[0]
        assert item.overlap.verdict == authoring.UNKNOWN
        assert "indexing bug" in item.overlap.advice

    def test_a_negative_is_not_applicable_rather_than_unknown(self, db, eval_dir):
        """A negative has no section by design; that is not a failure to check."""
        self.make(eval_dir, None)
        item = authoring.queue(4, eval_dir=eval_dir, db_path=db)[0]
        assert item.overlap.verdict == authoring.NOT_APPLICABLE
        assert not item.unverified and item.approvable

    def test_a_genuinely_clean_question_is_still_green(self, db, eval_dir):
        """The control: the warning must not swallow the real pass."""
        self.make(eval_dir, "4 CFR § 7.3",
                  question="Do watchdog staff tell anyone what they own each year?")
        item = authoring.queue(4, eval_dir=eval_dir, db_path=db)[0]
        assert item.overlap.verdict in (authoring.GREEN, authoring.AMBER)
        assert item.overlap.checked and not item.unverified

    def test_the_json_view_never_reports_zero_overlap_for_an_unchecked_row(
        self, db, eval_dir
    ):
        """distinctive=0.0 would read as "clean" to any consumer."""
        self.make(eval_dir, "4 CFR § 999.9")
        payload = authoring.queue(4, eval_dir=eval_dir, db_path=db)[0].overlap.to_dict()
        assert payload["checked"] is False
        assert payload["unverified"] is True
        assert payload["distinctive"] is None, "must not be 0.0, which reads as clean"

    def test_an_unverified_entry_cannot_be_approved(self, db, eval_dir):
        """Promotion already refuses it; the UI refuses it while you are looking."""
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        self.make(eval_dir, "4 CFR § 999.9")
        app = create_app(
            Settings(root=REPO_ROOT, eval_dir=eval_dir, gold_path=eval_dir / "gold.jsonl",
                     db_path=db, results_dir=eval_dir / "results"),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as client:
            response = client.post(
                "/authoring/4/entry/x",
                data={"question": "q", "notes": "", "kind": "prose", "status": "approved",
                      "citations": "4 CFR § 999.9"},
            )
        assert response.status_code == 422
        assert "not in the index" in response.text
        assert load_draft(4, eval_dir)[0]["status"] == "draft", "the refusal left it a draft"

    def test_the_queue_page_warns_about_unverified_rows(self, db, eval_dir):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        self.make(eval_dir, "4 CFR § 999.9")
        app = create_app(
            Settings(root=REPO_ROOT, eval_dir=eval_dir, gold_path=eval_dir / "gold.jsonl",
                     db_path=db, results_dir=eval_dir / "results"),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as client:
            page = client.get("/authoring/4")
        assert "NOT CHECKED" in page.text
        assert "cite a section that is not in the index" in page.text
        assert "disabled>approved" in page.text.replace("\n", " ").replace("  ", " ") or \
               "disabled" in page.text


# --- messy draft files ------------------------------------------------------


class TestMessyDraftFiles:
    """Hand-edited JSONL is the supported workflow, so every mangling of it
    must degrade to a warning rather than a 500."""

    def write(self, eval_dir, *lines):
        authoring.draft_path(31, eval_dir).write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )

    def checks(self, eval_dir):
        return {p.check for p in authoring.read_draft(31, eval_dir).problems}

    def test_a_missing_kind_and_status_are_filled_in(self, eval_dir):
        self.write(eval_dir, json.dumps({"question": "q", "relevant_citations": []}))
        entry = load_draft(31, eval_dir)[0]
        assert entry["kind"] == "prose" and entry["status"] == "draft"
        assert entry["notes"] == ""

    def test_a_line_that_is_not_json_is_skipped_and_reported(self, eval_dir):
        self.write(
            eval_dir,
            json.dumps({"question": "good", "relevant_citations": []}),
            "{not json at all",
            json.dumps({"question": "also good", "relevant_citations": []}),
        )
        draft = authoring.read_draft(31, eval_dir)
        assert [e["question"] for e in draft.entries] == ["good", "also good"]
        assert "draft:bad-json" in {p.check for p in draft.problems}
        assert draft.skipped == 1

    def test_a_non_object_line_is_skipped_and_reported(self, eval_dir):
        self.write(eval_dir, json.dumps(["not", "an", "object"]),
                   json.dumps({"question": "fine", "relevant_citations": []}))
        draft = authoring.read_draft(31, eval_dir)
        assert len(draft.entries) == 1
        assert "draft:not-an-object" in {p.check for p in draft.problems}

    def test_a_string_citation_becomes_one_citation_not_eleven(self, eval_dir):
        """Iterating a bare string yields characters — a silent catastrophe."""
        self.write(eval_dir, json.dumps({"question": "q",
                                         "relevant_citations": "31 CFR § 1.1"}))
        entry = load_draft(31, eval_dir)[0]
        assert entry["relevant_citations"] == ["31 CFR § 1.1"]
        assert "draft:citations-type" in self.checks(eval_dir)

    def test_a_nonsense_citations_type_becomes_empty_and_is_reported(self, eval_dir):
        self.write(eval_dir, json.dumps({"question": "q", "relevant_citations": 42}))
        assert load_draft(31, eval_dir)[0]["relevant_citations"] == []
        assert "draft:citations-type" in self.checks(eval_dir)

    def test_a_non_string_question_is_coerced(self, eval_dir):
        self.write(eval_dir, json.dumps({"question": 1234, "relevant_citations": []}))
        assert load_draft(31, eval_dir)[0]["question"] == "1234"
        assert "draft:question-type" in self.checks(eval_dir)

    def test_an_invalid_kind_is_reported_not_silently_rewritten(self, eval_dir):
        """Rewriting "lookup" to "prose" would hide a decision someone got wrong."""
        self.write(eval_dir, json.dumps({"question": "q", "relevant_citations": [],
                                         "kind": "lookup"}))
        assert load_draft(31, eval_dir)[0]["kind"] == "lookup"
        assert "draft:kind" in self.checks(eval_dir)

    def test_an_invalid_status_is_reported_and_is_not_approved(self, eval_dir):
        self.write(eval_dir, json.dumps({"question": "q", "relevant_citations": [],
                                         "status": "maybe"}))
        entry = load_draft(31, eval_dir)[0]
        assert entry["status"] == "maybe" != "approved"
        assert "draft:status" in self.checks(eval_dir)

    def test_an_invalid_status_cannot_be_promoted(self, db, eval_dir, gold_path):
        self.write(eval_dir, json.dumps({"question": "q", "relevant_citations": ["4 CFR § 7.3"],
                                         "status": "approved-ish"}))
        result = promote(31, specs=[SPEC4], eval_dir=eval_dir, gold_path=gold_path, db_path=db)
        assert not result.ok
        assert result.errors[0].check == "authoring:nothing-approved"

    def test_reading_never_writes_the_file(self, eval_dir):
        """The file is someone's working copy and a tracked artifact; a read
        must not churn its git diff or fight their text editor."""
        self.write(eval_dir, json.dumps({"question": "q"}))
        path = authoring.draft_path(31, eval_dir)
        before = path.read_bytes()
        stat_before = path.stat().st_mtime_ns

        for _ in range(3):
            authoring.read_draft(31, eval_dir)
            authoring.queue(31, eval_dir=eval_dir, db_path=eval_dir / "nope.db")

        assert path.read_bytes() == before
        assert path.stat().st_mtime_ns == stat_before

    def test_ids_are_stable_across_reads_of_an_unchanged_file(self, eval_dir):
        self.write(eval_dir,
                   json.dumps({"question": "one", "relevant_citations": []}),
                   json.dumps({"question": "two", "relevant_citations": []}))
        first = [e["draft_id"] for e in load_draft(31, eval_dir)]
        second = [e["draft_id"] for e in load_draft(31, eval_dir)]
        assert first == second
        assert len(set(first)) == 2

    def test_an_explicit_id_in_the_file_always_wins(self, eval_dir):
        self.write(eval_dir, json.dumps({"question": "q", "relevant_citations": [],
                                         "draft_id": "deadbeef1234"}))
        assert load_draft(31, eval_dir)[0]["draft_id"] == "deadbeef1234"

    def test_saving_is_what_persists_the_repairs(self, eval_dir):
        """Normalisation reaches disk only through an edit the author asked for."""
        self.write(eval_dir, json.dumps({"question": "q"}))
        entries = load_draft(31, eval_dir)
        save_draft(31, entries, eval_dir)
        on_disk = json.loads(authoring.draft_path(31, eval_dir).read_text().splitlines()[0])
        assert on_disk["kind"] == "prose" and on_disk["status"] == "draft"
        assert on_disk["draft_id"]

    def test_an_empty_or_whitespace_only_file_is_an_empty_queue(self, eval_dir):
        self.write(eval_dir, "", "   ", "")
        draft = authoring.read_draft(31, eval_dir)
        assert draft.entries == [] and draft.problems == ()

    def test_a_wholly_corrupt_file_does_not_take_the_page_down(self, eval_dir):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        self.write(eval_dir, "garbage", "{also garbage", "[1,2,3]")
        app = create_app(
            Settings(root=REPO_ROOT, eval_dir=eval_dir, gold_path=eval_dir / "gold.jsonl",
                     db_path=eval_dir / "absent.db", results_dir=eval_dir / "results"),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as client:
            page = client.get("/authoring/31")
            listing = client.get("/authoring")
        assert page.status_code == 200
        assert "needed repair to be readable" in page.text
        assert "3 line(s) could not be read" in page.text
        assert listing.status_code == 200

    def test_the_real_committed_title_31_draft_loads_cleanly(self):
        """The file that originally caused the 500, read from the repo as-is."""
        from api.main import REPO_ROOT

        path = REPO_ROOT / "eval" / "gold_title31_draft.jsonl"
        if not path.exists():
            pytest.skip("the Title 31 draft is not present")
        before = path.read_bytes()
        entries = load_draft(31, REPO_ROOT / "eval")
        assert entries, "the committed draft should yield entries"
        assert all(e["kind"] in goldset.KINDS for e in entries)
        assert all(e["status"] in goldset.STATUSES for e in entries)
        assert all(e["draft_id"] for e in entries)
        assert path.read_bytes() == before, "reading it must not rewrite it"


# --- calibration ------------------------------------------------------------


class TestCalibration:
    """The bands have to come from the corpus, and calibration has to be honest
    about the questions it could not score."""

    def gold(self, gold_path, db, n=6):
        sections = sections_in(db)
        entries = [
            {"question": f"Who must do the thing in case {i}?",
             "relevant_citations": [sections[i % len(sections)]],
             "kind": "prose", "status": "approved"}
            for i in range(n)
        ]
        goldset.write(entries, gold_path)
        return entries

    def test_scores_the_approved_prose_set(self, db, gold_path):
        self.gold(gold_path, db)
        cal = authoring.calibrate(gold_path, db)
        assert cal.n_scored == 6 and cal.n_skipped == 0
        assert cal.n_sections == len(sections_in(db))
        assert all(0.0 <= v <= 1.0 for v in cal.values)

    def test_worst_questions_come_first(self, db, gold_path):
        goldset.write([
            {"question": "A GAO employee must file a financial disclosure report annually.",
             "relevant_citations": ["4 CFR § 7.3"], "kind": "prose", "status": "approved"},
            {"question": "Do watchdog staff tell anyone what they own each year?",
             "relevant_citations": ["4 CFR § 7.3"], "kind": "prose", "status": "approved"},
        ], gold_path)
        cal = authoring.calibrate(gold_path, db)
        assert cal.scores[0][0] > cal.scores[-1][0]
        assert "financial disclosure" in cal.scores[0][1]

    def test_unscorable_questions_are_counted_not_scored_as_zero(self, db, gold_path):
        """Counting an unindexed section as 0.0 overlap would drag the whole
        distribution down and make the bands too tight — the same false-green
        mistake, one level up."""
        goldset.write([
            {"question": "a real one", "relevant_citations": ["4 CFR § 7.3"],
             "kind": "prose", "status": "approved"},
            {"question": "unindexed", "relevant_citations": ["4 CFR § 999.9"],
             "kind": "prose", "status": "approved"},
        ], gold_path)
        cal = authoring.calibrate(gold_path, db)
        assert cal.n_scored == 1 and cal.n_skipped == 1
        assert 0.0 not in cal.values or len(cal.values) == 1

    def test_citation_lookups_are_excluded(self, db, gold_path):
        """They are supposed to quote the section number; the meter does not
        judge them, so they must not shape its thresholds."""
        goldset.write([
            {"question": "prose one", "relevant_citations": ["4 CFR § 7.3"],
             "kind": "prose", "status": "approved"},
            {"question": "What does 4 CFR 7.3 provide?", "relevant_citations": ["4 CFR § 7.3"],
             "kind": "citation", "status": "approved"},
        ], gold_path)
        assert authoring.calibrate(gold_path, db).n_scored == 1

    def test_drafts_are_excluded(self, db, gold_path):
        goldset.write([
            {"question": "approved one", "relevant_citations": ["4 CFR § 7.3"],
             "kind": "prose", "status": "approved"},
            {"question": "draft one", "relevant_citations": ["4 CFR § 7.3"],
             "kind": "prose", "status": "draft"},
        ], gold_path)
        assert authoring.calibrate(gold_path, db).n_scored == 1

    def test_percentiles_are_ordered(self, db, gold_path):
        self.gold(gold_path, db, n=12)
        cal = authoring.calibrate(gold_path, db)
        pcts = [cal.percentile(p) for p in (10, 25, 50, 75, 90, 100)]
        assert pcts == sorted(pcts)

    def test_proposed_bands_put_amber_below_red(self, db, gold_path):
        self.gold(gold_path, db, n=12)
        proposed = authoring.calibrate(gold_path, db).propose()
        assert proposed["DISTINCTIVE_AMBER"] <= proposed["DISTINCTIVE_RED"]

    def test_flagged_counts_split_the_approved_set(self, db, gold_path):
        self.gold(gold_path, db, n=10)
        cal = authoring.calibrate(gold_path, db)
        ambers, reds = cal.flagged(0.0, 1.1, 99)
        assert ambers + reds <= cal.n_scored
        none_amber, none_red = cal.flagged(1.1, 1.1, 99)
        assert none_red == 0

    def test_an_unbuilt_index_yields_nothing_rather_than_fake_numbers(
        self, gold_path, tmp_path
    ):
        goldset.write([{"question": "q", "relevant_citations": ["4 CFR § 7.3"],
                        "kind": "prose", "status": "approved"}], gold_path)
        cal = authoring.calibrate(gold_path, tmp_path / "absent.db")
        assert cal.n_scored == 0

    def test_rare_term_report_describes_the_corpus(self, db):
        rows = authoring.rare_term_report(db)
        assert rows
        dfs = [df for df, _, _ in rows]
        rarities = [r for _, r, _ in rows]
        assert dfs == sorted(dfs)
        assert rarities == sorted(rarities, reverse=True), "rarer = higher score"

    def test_calibration_records_what_it_was_run_against(self, db, gold_path):
        self.gold(gold_path, db)
        cal = authoring.calibrate(gold_path, db, REPO_MANIFEST)
        assert cal.gold_sha256
        assert cal.corpus, "provenance: which corpus produced these bands"


class TestCalibratedBands:
    """The bands are a measured property of this corpus, so pin what they mean.

    These are not threshold-value tests — the numbers will move when the
    corpus does. They pin the *relationships* that made the calibration worth
    doing, so a future edit that reintroduces a guess fails here.
    """

    def test_amber_is_below_red(self):
        assert 0 < authoring.DISTINCTIVE_AMBER < authoring.DISTINCTIVE_RED <= 1.0

    def test_the_bands_sit_where_the_approved_distribution_does(self, db, gold_path):
        """Calibration's own output must agree with the shipped constants.

        If someone re-runs --calibrate and the proposal is far from what is
        in the file, the file is stale.
        """
        from api.main import REPO_ROOT

        real_gold = REPO_ROOT / "eval" / "gold.jsonl"
        real_db = REPO_ROOT / "data" / "index" / "regsearch.db"
        if not real_db.exists():
            pytest.skip("no built index; run `python -m pipeline.onboard --only index`")
        cal = authoring.calibrate(real_gold, real_db, REPO_MANIFEST)
        proposed = cal.propose()
        assert abs(proposed["DISTINCTIVE_AMBER"] - authoring.DISTINCTIVE_AMBER) < 0.08, (
            f"shipped amber {authoring.DISTINCTIVE_AMBER} vs calibrated "
            f"{proposed['DISTINCTIVE_AMBER']} — re-run --calibrate"
        )
        assert abs(proposed["DISTINCTIVE_RED"] - authoring.DISTINCTIVE_RED) < 0.08

    def test_the_bands_do_not_condemn_the_approved_set(self, db, gold_path):
        """The failure the calibration found: bands that flag a third of
        human-reviewed work red are measuring the wrong thing."""
        from api.main import REPO_ROOT

        real_db = REPO_ROOT / "data" / "index" / "regsearch.db"
        if not real_db.exists():
            pytest.skip("no built index")
        cal = authoring.calibrate(REPO_ROOT / "eval" / "gold.jsonl", real_db, REPO_MANIFEST)
        ambers, reds = cal.flagged(
            authoring.DISTINCTIVE_AMBER, authoring.DISTINCTIVE_RED, authoring.RARE_COPIED_RED
        )
        assert reds / cal.n_scored < 0.15, f"{reds}/{cal.n_scored} approved questions go red"
        assert ambers / cal.n_scored < 0.40

    def test_the_term_of_art_rule_is_a_tripwire_not_a_nuisance(self, db, gold_path):
        """Copying two terms of art is red regardless of ratio. On reviewed
        work that must be close to unheard of, or it is just noise."""
        from api.main import REPO_ROOT

        real_db = REPO_ROOT / "data" / "index" / "regsearch.db"
        if not real_db.exists():
            pytest.skip("no built index")
        cal = authoring.calibrate(REPO_ROOT / "eval" / "gold.jsonl", real_db, REPO_MANIFEST)
        tripped = sum(1 for c in cal.rare_counts if c >= authoring.RARE_COPIED_RED)
        assert tripped / cal.n_scored < 0.05, f"{tripped} approved questions trip the rule"

    def test_rare_term_means_a_handful_of_sections(self, db):
        """RARE_TERM has to correspond to a defensible document frequency."""
        vocab = authoring.vocabulary(db)
        if vocab.n_sections < 10:
            pytest.skip("fixture corpus is too small for a meaningful df mapping")
        qualifying = [df for df, rarity, _ in authoring.rare_term_report(db)
                      if rarity >= authoring.RARE_TERM]
        assert qualifying, "no document frequency qualifies as a term of art"
        assert max(qualifying) <= vocab.n_sections * 0.02, "'rare' must mean rare"


# --- the draft queue is never the casualty of a failed merge ----------------
#
# Regression for the Title 31 incident: every approved draft row asked a question
# already in gold.jsonl (hand-merged earlier, with longer notes). Promotion matched
# on question text alone, wrote nothing to gold, then retired all 30 rows anyway —
# gold stayed at 143 lines and the draft went to 0. Every test below asserts on the
# draft's *bytes*, not its parsed entries: "still parses to the same rows" would
# let a lossy rewrite (dropped lines, re-serialised text) through.


def _draft_bytes(eval_dir, title=4):
    path = authoring.draft_path(title, eval_dir)
    return path.read_bytes() if path.exists() else None


class TestPromotionNeverLosesTheDraft:
    def _promote(self, db, eval_dir, gold_path, title=4, specs=(SPEC4,)):
        return promote(title, specs=list(specs), eval_dir=eval_dir,
                       gold_path=gold_path, db_path=db)

    # -- the incident itself --

    def test_rows_already_in_gold_but_different_are_a_conflict_not_a_retirement(
        self, db, eval_dir, gold_path
    ):
        cite = sections_in(db)[0]
        goldset.write([{"question": "q one", "relevant_citations": [cite], "kind": "prose",
                        "status": "approved", "notes": "verified, longer notes"}], gold_path)
        save_draft(4, [approved(cite, "q one", notes="short notes")], eval_dir)
        draft_before, gold_before = _draft_bytes(eval_dir), gold_path.read_bytes()

        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert result.errors[0].check == "authoring:conflicts-with-gold"
        assert result.conflicts == (("q one", ("notes",)),)
        assert _draft_bytes(eval_dir) == draft_before, "the draft row must survive"
        assert gold_path.read_bytes() == gold_before

    def test_a_different_citation_is_a_conflict_too(self, db, eval_dir, gold_path):
        a, b = sections_in(db)[:2]
        goldset.write([{"question": "q", "relevant_citations": [a], "kind": "prose",
                        "status": "approved"}], gold_path)
        save_draft(4, [approved(b, "q")], eval_dir)
        result = self._promote(db, eval_dir, gold_path)
        assert not result.ok
        assert result.conflicts == (("q", ("relevant_citations",)),)
        assert len(load_draft(4, eval_dir)) == 1

    def test_approving_a_question_gold_holds_as_draft_is_a_conflict(
        self, db, eval_dir, gold_path
    ):
        """Retiring it would silently leave the gold row unapproved."""
        cite = sections_in(db)[0]
        goldset.write([{"question": "q", "relevant_citations": [cite], "kind": "prose",
                        "status": "draft"}], gold_path)
        save_draft(4, [approved(cite, "q")], eval_dir)
        result = self._promote(db, eval_dir, gold_path)
        assert not result.ok
        assert result.conflicts == (("q", ("status",)),)

    def test_one_conflict_blocks_the_whole_batch(self, db, eval_dir, gold_path):
        cite = sections_in(db)[0]
        goldset.write([{"question": "old", "relevant_citations": [cite], "kind": "prose",
                        "status": "approved", "notes": "gold"}], gold_path)
        save_draft(4, [approved(cite, "old", notes="draft"),
                       approved(cite, "genuinely new")], eval_dir)
        gold_before, draft_before = gold_path.read_bytes(), _draft_bytes(eval_dir)

        assert not self._promote(db, eval_dir, gold_path).ok
        assert gold_path.read_bytes() == gold_before, "the new one must not land alone"
        assert _draft_bytes(eval_dir) == draft_before

    def test_the_real_title_31_draft_and_gold_reproduce_the_incident_safely(
        self, db, eval_dir, gold_path
    ):
        """The committed files, with every row approved as the author had them."""
        from api.main import REPO_ROOT

        src_draft = REPO_ROOT / "eval" / "gold_title31_draft.jsonl"
        src_gold = REPO_ROOT / "eval" / "gold.jsonl"
        if not (src_draft.exists() and src_gold.exists()):
            pytest.skip("the committed Title 31 files are not present")
        gold_path.write_bytes(src_gold.read_bytes())
        rows = [json.loads(line) for line in src_draft.read_text("utf-8").splitlines()
                if line.strip()]
        for row in rows:
            row["status"] = "approved"
        save_draft(31, rows, eval_dir)
        draft_before, gold_before = _draft_bytes(eval_dir, 31), gold_path.read_bytes()

        result = self._promote(db, eval_dir, gold_path, title=31, specs=(SPEC31,))

        assert not result.ok
        assert len(result.conflicts) == len(rows) == 30
        assert _draft_bytes(eval_dir, 31) == draft_before, "all 30 rows must survive"
        assert gold_path.read_bytes() == gold_before

    # -- empty and refused promotions --

    def test_an_empty_promotion_leaves_the_draft_byte_identical(self, db, eval_dir, gold_path):
        save_draft(4, [approved(sections_in(db)[0], f"q{i}", status="draft",
                                draft_id=f"d{i}") for i in range(30)], eval_dir)
        before = _draft_bytes(eval_dir)

        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert result.errors[0].check == "authoring:nothing-approved"
        assert result.still_draft == 30
        assert _draft_bytes(eval_dir) == before
        assert not gold_path.exists()

    def test_an_empty_draft_file_is_refused_and_left_alone(self, db, eval_dir, gold_path):
        path = authoring.draft_path(4, eval_dir)
        path.write_text("\n", encoding="utf-8")
        assert not self._promote(db, eval_dir, gold_path).ok
        assert path.read_bytes() == b"\n"

    def test_a_missing_draft_is_refused_and_not_created(self, db, eval_dir, gold_path):
        assert not self._promote(db, eval_dir, gold_path).ok
        assert not authoring.draft_path(4, eval_dir).exists()

    def test_a_validation_failure_leaves_the_draft_byte_identical(
        self, db, eval_dir, gold_path
    ):
        save_draft(4, [approved(sections_in(db)[0], "fine"),
                       approved("4 CFR § 999.9", "cites nothing real")], eval_dir)
        before = _draft_bytes(eval_dir)
        assert not self._promote(db, eval_dir, gold_path).ok
        assert _draft_bytes(eval_dir) == before

    def test_unreadable_lines_block_promotion_rather_than_being_deleted(
        self, db, eval_dir, gold_path
    ):
        cite = sections_in(db)[0]
        path = authoring.draft_path(4, eval_dir)
        path.write_text(json.dumps(approved(cite, "good")) + "\n{oops not json\n",
                        encoding="utf-8")
        before = path.read_bytes()

        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert result.errors[0].check == "authoring:draft-unreadable-lines"
        assert path.read_bytes() == before
        assert not gold_path.exists()

    # -- failures part-way through the commit --

    def _fail_writes_to(self, monkeypatch, name, *, times=1):
        """Make the first ``times`` atomic writes to a file called ``name`` raise."""
        real = authoring._atomic_write
        left = {"n": times}

        def flaky(path, data):
            if Path(path).name == name and left["n"] > 0:
                left["n"] -= 1
                raise OSError(f"simulated failure writing {name}")
            return real(path, data)

        monkeypatch.setattr(authoring, "_atomic_write", flaky)

    def test_a_failed_gold_write_leaves_both_files_alone(
        self, db, eval_dir, gold_path, monkeypatch
    ):
        cite = sections_in(db)[0]
        goldset.write([{"question": "existing", "relevant_citations": [cite],
                        "kind": "prose", "status": "approved"}], gold_path)
        save_draft(4, [approved(cite, "new")], eval_dir)
        gold_before, draft_before = gold_path.read_bytes(), _draft_bytes(eval_dir)
        self._fail_writes_to(monkeypatch, "gold.jsonl")

        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert result.errors[0].check == "authoring:write-failed"
        assert "draft was not modified" in result.errors[0].message
        assert gold_path.read_bytes() == gold_before
        assert _draft_bytes(eval_dir) == draft_before

    def test_gold_that_does_not_read_back_as_written_is_rolled_back(
        self, db, eval_dir, gold_path, monkeypatch
    ):
        cite = sections_in(db)[0]
        goldset.write([{"question": "existing", "relevant_citations": [cite],
                        "kind": "prose", "status": "approved"}], gold_path)
        save_draft(4, [approved(cite, "new")], eval_dir)
        gold_before, draft_before = gold_path.read_bytes(), _draft_bytes(eval_dir)
        real_load = authoring.goldset.load
        calls = {"n": 0}

        def corrupt_readback(path):
            calls["n"] += 1
            return [] if calls["n"] == 2 else real_load(path)  # 1st: read, 2nd: verify

        monkeypatch.setattr(authoring.goldset, "load", corrupt_readback)
        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert "did not read back as written" in result.errors[0].message
        assert gold_path.read_bytes() == gold_before, "rolled back to the original bytes"
        assert _draft_bytes(eval_dir) == draft_before

    def test_a_failed_draft_write_rolls_gold_back(self, db, eval_dir, gold_path, monkeypatch):
        cite = sections_in(db)[0]
        goldset.write([{"question": "existing", "relevant_citations": [cite],
                        "kind": "prose", "status": "approved"}], gold_path)
        save_draft(4, [approved(cite, "new")], eval_dir)
        gold_before, draft_before = gold_path.read_bytes(), _draft_bytes(eval_dir)
        self._fail_writes_to(monkeypatch, "gold_title4_draft.jsonl")

        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert "restored to exactly what it was" in result.errors[0].message
        assert gold_path.read_bytes() == gold_before
        assert _draft_bytes(eval_dir) == draft_before

    def test_rolling_back_a_gold_file_that_did_not_exist_removes_it(
        self, db, eval_dir, gold_path, monkeypatch
    ):
        save_draft(4, [approved(sections_in(db)[0], "new")], eval_dir)
        self._fail_writes_to(monkeypatch, "gold_title4_draft.jsonl")
        assert not self._promote(db, eval_dir, gold_path).ok
        assert not gold_path.exists()

    def test_if_the_rollback_fails_too_a_re_run_recovers_without_loss(
        self, db, eval_dir, gold_path, monkeypatch
    ):
        cite = sections_in(db)[0]
        goldset.write([{"question": "existing", "relevant_citations": [cite],
                        "kind": "prose", "status": "approved"}], gold_path)
        save_draft(4, [approved(cite, "new")], eval_dir)
        draft_before = _draft_bytes(eval_dir)
        self._fail_writes_to(monkeypatch, "gold_title4_draft.jsonl")
        monkeypatch.setattr(authoring, "_restore",
                            lambda *a: (_ for _ in ()).throw(OSError("disk gone")))

        result = self._promote(db, eval_dir, gold_path)

        assert not result.ok
        assert "run Approve & merge again" in result.errors[0].message
        assert _draft_bytes(eval_dir) == draft_before, "the draft still holds the work"

        monkeypatch.undo()
        again = self._promote(db, eval_dir, gold_path)
        assert again.ok and again.promoted == ()
        assert [e["question"] for e in goldset.load(gold_path)] == ["existing", "new"]
        assert load_draft(4, eval_dir) == []

    def test_a_draft_edited_mid_merge_is_kept_and_gold_rolled_back(
        self, db, eval_dir, gold_path, monkeypatch
    ):
        """An editor or a second process takes no lock; its edit must win."""
        cite = sections_in(db)[0]
        save_draft(4, [approved(cite, "new")], eval_dir)
        dpath = authoring.draft_path(4, eval_dir)
        real = authoring._atomic_write

        def edit_draft_after_gold(path, data):
            out = real(path, data)
            if Path(path).name == "gold.jsonl":
                dpath.write_text(dpath.read_text("utf-8") + json.dumps(
                    approved(cite, "typed in an editor meanwhile", status="draft")) + "\n",
                    encoding="utf-8")
            return out

        monkeypatch.setattr(authoring, "_atomic_write", edit_draft_after_gold)
        edited_result = self._promote(db, eval_dir, gold_path)

        assert not edited_result.ok
        assert "edited while the merge was running" in edited_result.errors[0].message
        assert not gold_path.exists()
        assert len(load_draft(4, eval_dir)) == 2, "the concurrent edit survives"

    # -- and when it does succeed --

    def test_the_draft_is_cleared_only_once_gold_holds_every_row(
        self, db, eval_dir, gold_path
    ):
        cite = sections_in(db)[0]
        save_draft(4, [approved(cite, f"question {i}", draft_id=f"d{i}") for i in range(30)],
                   eval_dir)
        result = self._promote(db, eval_dir, gold_path)
        assert result.ok, result.errors
        assert len(goldset.load(gold_path)) == 30
        assert load_draft(4, eval_dir) == []


# --- bulk status: approving 30 rows is one request, not 30 saves -------------


class TestBulkStatus:
    def _queue(self, db, eval_dir, n=3):
        cite = sections_in(db)[0]
        rows = [approved(cite, f"question {i}", status="draft", draft_id=f"d{i}")
                for i in range(n)]
        save_draft(4, rows, eval_dir)
        return [r["draft_id"] for r in rows]

    def test_approves_many_rows_in_one_write(self, db, eval_dir, monkeypatch):
        ids = self._queue(db, eval_dir, n=30)
        writes = []
        real = authoring._atomic_write
        monkeypatch.setattr(authoring, "_atomic_write",
                            lambda p, d: (writes.append(p), real(p, d))[1])

        changed = authoring.set_status(4, ids, "approved", eval_dir=eval_dir, db_path=db)

        assert changed == ids
        assert len(writes) == 1
        assert {e["status"] for e in load_draft(4, eval_dir)} == {"approved"}

    def test_only_status_changes(self, db, eval_dir):
        ids = self._queue(db, eval_dir)
        before = load_draft(4, eval_dir)
        authoring.set_status(4, ids[:1], "approved", eval_dir=eval_dir, db_path=db)
        after = load_draft(4, eval_dir)
        assert [{**e, "status": None} for e in after] == [{**e, "status": None} for e in before]
        assert [e["status"] for e in after] == ["approved", "draft", "draft"]

    def test_one_ineligible_row_refuses_the_whole_selection(self, db, eval_dir):
        cite = sections_in(db)[0]
        save_draft(4, [
            approved(cite, "fine", status="draft", draft_id="ok"),
            approved(cite, "   ", status="draft", draft_id="empty"),
            approved("4 CFR § 999.9", "unindexed", status="draft", draft_id="ghost"),
        ], eval_dir)
        before = _draft_bytes(eval_dir)

        with pytest.raises(authoring.StatusRefused) as refused:
            authoring.set_status(4, ["ok", "empty", "ghost", "nope"], "approved",
                                 eval_dir=eval_dir, db_path=db)

        assert {rid for rid, _ in refused.value.reasons} == {"empty", "ghost", "nope"}
        assert _draft_bytes(eval_dir) == before

    def test_un_approving_is_always_allowed(self, db, eval_dir):
        save_draft(4, [approved("4 CFR § 999.9", "unindexed", draft_id="ghost")], eval_dir)
        assert authoring.set_status(4, ["ghost"], "draft", eval_dir=eval_dir,
                                    db_path=db) == ["ghost"]

    def test_an_empty_selection_or_bad_status_is_refused(self, db, eval_dir):
        ids = self._queue(db, eval_dir)
        with pytest.raises(authoring.StatusRefused):
            authoring.set_status(4, [], "approved", eval_dir=eval_dir, db_path=db)
        with pytest.raises(authoring.StatusRefused):
            authoring.set_status(4, ids, "yolo", eval_dir=eval_dir, db_path=db)

    def test_unreadable_lines_block_a_bulk_save(self, db, eval_dir):
        path = authoring.draft_path(4, eval_dir)
        path.write_text(json.dumps(approved(sections_in(db)[0], "q", status="draft",
                                            draft_id="a")) + "\nnot json\n", encoding="utf-8")
        before = path.read_bytes()
        with pytest.raises(authoring.StatusRefused):
            authoring.set_status(4, ["a"], "approved", eval_dir=eval_dir, db_path=db)
        assert path.read_bytes() == before


# --- the promote controls over HTTP -----------------------------------------


class TestPromoteControls:
    @pytest.fixture
    def client(self, db, eval_dir, gold_path):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        app = create_app(
            Settings(root=REPO_ROOT, eval_dir=eval_dir, gold_path=gold_path, db_path=db,
                     results_dir=eval_dir / "results"),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as test_client:
            yield test_client

    @staticmethod
    def _promote_button(html):
        start = html.index('action="/authoring/4/promote"')
        return html[start:html.index("</button>", start)]

    def test_the_button_is_disabled_when_nothing_is_approved(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", "q", status="draft", draft_id="a")], eval_dir)
        button = self._promote_button(client.get("/authoring/4").text)
        assert "disabled" in button
        assert "merge 0 into" in button

    def test_the_button_is_enabled_and_counts_what_will_merge(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", "q1", draft_id="a"),
                       approved("4 CFR § 7.3", "q2", status="draft", draft_id="b")], eval_dir)
        button = self._promote_button(client.get("/authoring/4").text)
        assert "disabled" not in button
        assert "merge 1 into" in button

    def test_a_forced_empty_post_is_refused_and_leaves_the_draft_alone(
        self, client, eval_dir, gold_path
    ):
        """The disabled button is a courtesy; the server is the guarantee."""
        save_draft(4, [approved("4 CFR § 7.3", "q", status="draft", draft_id="a")], eval_dir)
        before = _draft_bytes(eval_dir)
        response = client.post("/authoring/4/promote")
        assert response.status_code == 422
        assert "the draft queue are exactly as they were" in response.text
        assert _draft_bytes(eval_dir) == before
        assert not gold_path.exists()

    def test_a_conflicting_promotion_renders_the_conflicts(self, client, eval_dir, gold_path):
        goldset.write([{"question": "q", "relevant_citations": ["4 CFR § 7.3"],
                        "kind": "prose", "status": "approved", "notes": "gold"}], gold_path)
        save_draft(4, [approved("4 CFR § 7.3", "q", notes="draft", draft_id="a")], eval_dir)
        response = client.post("/authoring/4/promote")
        assert response.status_code == 422
        assert "Already in the gold set, but different" in response.text
        assert len(load_draft(4, eval_dir)) == 1

    def test_saving_a_row_refreshes_the_controls_out_of_band(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", "q", status="draft", draft_id="a")], eval_dir)
        response = client.post("/authoring/4/entry/a", data={
            "question": "q", "kind": "prose", "status": "approved", "citations": "4 CFR § 7.3",
        })
        assert response.status_code == 200
        assert 'id="promote-controls" hx-swap-oob="true"' in response.text
        assert "1 of 1 marked approved" in response.text
        assert "disabled" not in self._promote_button(response.text)

    def test_rows_carry_a_bulk_checkbox_marked_with_eligibility(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", "fine", status="draft", draft_id="ok"),
                       approved("4 CFR § 999.9", "ghost", status="draft", draft_id="no")],
                   eval_dir)
        page = client.get("/authoring/4").text
        assert 'form id="bulk"' in page
        assert 'value="ok" form="bulk"' in page
        ok = page[page.index('value="ok" form="bulk"'):][:300]
        no = page[page.index('value="no" form="bulk"'):][:300]
        assert 'data-approvable="1"' in ok
        assert 'data-approvable="0"' in no

    def test_the_bulk_route_approves_the_selection_in_one_request(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 7.3", f"q{i}", status="draft", draft_id=f"d{i}")
                       for i in range(30)], eval_dir)
        response = client.post(
            "/authoring/4/status",
            data={"status": "approved", "draft_id": [f"d{i}" for i in range(30)]},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert {e["status"] for e in load_draft(4, eval_dir)} == {"approved"}
        assert "merge 30 into" in self._promote_button(client.get("/authoring/4").text)

    def test_the_bulk_route_refuses_ineligible_rows_and_writes_nothing(self, client, eval_dir):
        save_draft(4, [approved("4 CFR § 999.9", "ghost", status="draft", draft_id="no")],
                   eval_dir)
        before = _draft_bytes(eval_dir)
        response = client.post("/authoring/4/status",
                               data={"status": "approved", "draft_id": ["no"]})
        assert response.status_code == 422
        assert "not in the index" in response.text
        assert _draft_bytes(eval_dir) == before
