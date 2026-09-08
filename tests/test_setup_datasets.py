"""Unit tests for the fuzzy transform and its matching query generators.

Focus is on the non-obvious behavior that would break the benchmark:

  - each Levenshtein edit type must transform length correctly and (for
    substitute) must actually change a character
  - the ``len < 2`` early return in ``_apply_fuzzy_edit`` (edge case)
  - the ``fuzzy`` transform must produce a stable base word for variant 0 and a
    variant within ``target_distance`` edits otherwise
  - the whole transform must be deterministic across runs (benchmarks depend
    on reproducibility)
  - the ``fuzzy`` query generator must produce the exact same base words that
    ``fuzzy`` transform emits as variant 0 (otherwise queries wouldn't match
    any dataset row)
  - the ``tag_only`` query generator must rotate through the tag list
"""

import csv
import random
import sys
from pathlib import Path

# Ensure scripts/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import setup_datasets  # noqa: E402


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,  # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + (ca != cb),  # substitution
                )
            )
        prev = curr
    return prev[-1]


# ---- _apply_fuzzy_edit ------------------------------------------------------


class TestApplyFuzzyEdit:
    def test_insert_increases_length_by_one(self):
        assert (
            len(setup_datasets._apply_fuzzy_edit("hello", "insert", random.Random(1)))
            == 6
        )

    def test_delete_decreases_length_by_one(self):
        assert (
            len(setup_datasets._apply_fuzzy_edit("hello", "delete", random.Random(1)))
            == 4
        )

    def test_substitute_preserves_length_and_changes_a_character(self):
        """Substitute must retry until a different character is produced."""
        result = setup_datasets._apply_fuzzy_edit(
            "hello", "substitute", random.Random(1)
        )
        assert len(result) == 5
        assert result != "hello"

    def test_short_word_returned_unchanged(self):
        """len < 2 must short-circuit and return the input verbatim."""
        assert setup_datasets._apply_fuzzy_edit("a", "delete", random.Random(1)) == "a"


# ---- fuzzy transform -------------------------------------------------------


def _fuzzy_transform(**overrides):
    t = {
        "type": "fuzzy",
        "variant_count": 5,
        "docs_per_variant": 2,
        "term_count": 10,
        "min_word_length": 8,
        "max_word_length": 8,
        "target_distance": 1,
    }
    t.update(overrides)
    return t


class TestFuzzyTransform:
    def test_all_docs_in_variant_zero_share_the_same_base_word(self):
        """Every doc that maps to variant 0 of a term must be identical.

        This is the contract that makes the fuzzy query type meaningful.
        """
        t = _fuzzy_transform()  # docs_per_variant=2 → doc 1 and doc 2 are v0/term1
        doc1 = setup_datasets.apply_transforms("", [t], 100, 1, 100)
        doc2 = setup_datasets.apply_transforms("", [t], 100, 2, 100)
        assert doc1 == doc2

    def test_variant_one_differs_from_base_by_exactly_target_distance(self):
        """target_distance=1 must give exactly 1 Levenshtein edit."""
        t = _fuzzy_transform(target_distance=1)
        base = setup_datasets.apply_transforms("", [t], 100, 1, 100)
        variant = setup_datasets.apply_transforms("", [t], 100, 3, 100)
        assert variant != base
        assert _levenshtein(base, variant) == 1

    def test_variant_distance_bounded_by_target_distance(self):
        """target_distance=3 must produce at most 3 edits (edits may compose)."""
        t = _fuzzy_transform(target_distance=3)
        base = setup_datasets.apply_transforms("", [t], 100, 1, 100)
        variant = setup_datasets.apply_transforms("", [t], 100, 3, 100)
        assert 1 <= _levenshtein(base, variant) <= 3

    def test_transform_is_deterministic_across_invocations(self):
        """Benchmark reproducibility depends on stable output for same inputs."""
        t = _fuzzy_transform()
        for doc_num in [1, 5, 17, 42]:
            a = setup_datasets.apply_transforms("", [t], 100, doc_num, 100)
            b = setup_datasets.apply_transforms("", [t], 100, doc_num, 100)
            assert a == b, f"non-deterministic for doc {doc_num}"


# ---- Query generation ------------------------------------------------------


class TestGenerateFuzzyQueries:
    def test_query_terms_equal_dataset_variant_zero(self, tmp_path: Path):
        """Each query term must equal the base word for that term_id in the dataset.

        If this contract breaks, fuzzy queries will not match anything.
        """
        query_config = {
            "type": "fuzzy",
            "doc_count": 5,
            "min_word_length": 8,
            "max_word_length": 8,
        }
        setup_datasets.generate_queries(tmp_path, query_config, "fuzzy_q.csv")

        with open(tmp_path / "fuzzy_q.csv") as f:
            reader = csv.reader(f)
            assert next(reader) == ["term"]
            query_terms = [row[0] for row in reader]

        assert len(query_terms) == 5

        dataset_transform = _fuzzy_transform(
            variant_count=5, docs_per_variant=1, term_count=5
        )
        # docs_per_variant=1, variant_count=5 → doc 1 = term 1 v0,
        # doc 6 = term 2 v0, doc 11 = term 3 v0, ...
        for i, expected_term in enumerate(query_terms, start=1):
            doc_num_for_v0 = (i - 1) * 5 + 1
            dataset_v0 = setup_datasets.apply_transforms(
                "", [dataset_transform], 100, doc_num_for_v0, 100
            )
            assert dataset_v0 == expected_term


class TestGenerateTagOnlyQueries:
    def test_rotates_through_tag_list_with_category_header(self, tmp_path: Path):
        query_config = {
            "type": "tag_only",
            "doc_count": 7,
            "tags": ["electronics", "books", "clothing"],
        }
        setup_datasets.generate_queries(tmp_path, query_config, "tag_q.csv")

        with open(tmp_path / "tag_q.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row[0] for row in reader]

        assert header == ["category"]
        assert rows == [
            "electronics",
            "books",
            "clothing",
            "electronics",
            "books",
            "clothing",
            "electronics",
        ]


# ---- vector NPY: dataset ↔ query token alignment ---------------------------


class TestVectorHybridQueryAlignment:
    """The vector benchmark's ingest side stores ``phrase{qid}`` per doc, and
    the vector query side emits ``phrase{i}`` as its ``search_term``. If either
    side changes format (say, ``phrase_{i}`` or ``phrase-{i}``), Group 15 will
    silently return zero results and the whole KNN + text-prefilter benchmark
    becomes meaningless. This is the same class of fragile contract that
    ``TestGenerateFuzzyQueries`` guards for fuzzy.
    """

    def test_every_query_term_appears_in_the_hybrid_dataset(self, tmp_path: Path):
        import numpy as np

        dims = 4
        doc_count = 10
        repeats = 2  # → 5 distinct phrase ids: phrase0..phrase4
        num_queries = 5

        dataset_config = {
            "doc_count": doc_count,
            "fields": [
                {
                    "name": "title",
                    "size": 50,
                    "transforms": [
                        {
                            "type": "proximity_phrase",
                            "term_count": 1,
                            "combinations": 1,
                            "repeats": repeats,
                        }
                    ],
                },
                {
                    "name": "embedding",
                    "size": 1,
                    "transforms": [{"type": "vector", "dimensions": dims}],
                },
            ],
        }
        setup_datasets.generate_structured_npy(tmp_path, "hybrid.npy", dataset_config)

        query_config = {"type": "vector", "doc_count": num_queries, "dimensions": dims}
        setup_datasets.generate_queries(tmp_path, query_config, "queries.csv")

        hybrid = np.load(tmp_path / "hybrid.npy", allow_pickle=False)
        queries = np.load(tmp_path / "queries.npy", allow_pickle=False)

        titles = {t.rstrip(b"\x00").decode("ascii") for t in hybrid["title"]}
        for term in queries["search_term"]:
            decoded = term.rstrip(b"\x00").decode("ascii")
            assert decoded in titles, (
                f"query term {decoded!r} does not appear in hybrid dataset titles "
                f"({sorted(titles)}) — KNN benchmark would silently return no hits"
            )


# ---- _int_to_base26 --------------------------------------------------------


class TestIntToBase26:
    """Base-26 encoding backs unique_tokens / cyclic_pattern / progressive_prefix.
    An off-by-one in the carry would collide tokens that must stay distinct.
    """

    def test_single_letter_range(self):
        assert setup_datasets._int_to_base26(0) == "a"
        assert setup_datasets._int_to_base26(25) == "z"

    def test_two_letter_rollover(self):
        # 26 must roll to "aa" (not "ba"): the sequence is a..z, aa, ab, ...
        assert setup_datasets._int_to_base26(26) == "aa"
        assert setup_datasets._int_to_base26(27) == "ab"
        assert setup_datasets._int_to_base26(51) == "az"
        assert setup_datasets._int_to_base26(52) == "ba"

    def test_three_letter_rollover(self):
        assert setup_datasets._int_to_base26(701) == "zz"
        assert setup_datasets._int_to_base26(702) == "aaa"

    def test_values_are_unique_and_monotonic_in_length(self):
        seen = [setup_datasets._int_to_base26(i) for i in range(1000)]
        assert len(set(seen)) == 1000  # no collisions
        # length is non-decreasing as the integer grows
        lengths = [len(s) for s in seen]
        assert lengths == sorted(lengths)


# ---- ingestion transforms --------------------------------------------------


def _apply(t, field_size=1000, doc_num=1, total_docs=100):
    return setup_datasets.apply_transforms("", [t], field_size, doc_num, total_docs)


class TestRepeatedToken:
    def test_emits_token_repeated_n_times(self):
        content = _apply({"type": "repeated_token", "token": "b", "token_count": 5})
        assert content == "b b b b b"

    def test_deterministic(self):
        t = {"type": "repeated_token", "token": "z", "token_count": 8}
        assert _apply(t) == _apply(t)


class TestCyclicPattern:
    def test_cycles_through_alphabet(self):
        content = _apply(
            {"type": "cyclic_pattern", "cycle_length": 3, "token_count": 7}
        )
        assert content.split() == ["a", "b", "c", "a", "b", "c", "a"]

    def test_deterministic(self):
        t = {"type": "cyclic_pattern", "cycle_length": 5, "token_count": 20}
        assert _apply(t) == _apply(t)


class TestUniqueTokens:
    def test_tokens_are_disjoint_across_documents(self):
        t = {"type": "unique_tokens", "token_count": 4}
        doc1 = set(_apply(t, doc_num=1).split())
        doc2 = set(_apply(t, doc_num=2).split())
        assert doc1 == {"a", "b", "c", "d"}
        assert doc2 == {"e", "f", "g", "h"}
        assert doc1.isdisjoint(doc2)

    def test_deterministic(self):
        t = {"type": "unique_tokens", "token_count": 10}
        assert _apply(t, doc_num=3) == _apply(t, doc_num=3)


class TestUuidTokens:
    def test_token_count_and_length(self):
        content = _apply(
            {"type": "uuid_tokens", "token_count": 3, "char_length": 8},
            field_size=1000,
        )
        tokens = content.split()
        assert len(tokens) == 3
        assert all(len(tok) == 8 for tok in tokens)

    def test_seeded_and_reproducible_per_doc(self):
        t = {"type": "uuid_tokens", "token_count": 3, "char_length": 16}
        assert _apply(t, doc_num=7) == _apply(t, doc_num=7)
        assert _apply(t, doc_num=7) != _apply(t, doc_num=8)


class TestProgressivePrefix:
    def test_prefixes_grow_in_length(self):
        content = _apply(
            {"type": "progressive_prefix", "max_depth": 3, "leaf_count": 2},
            doc_num=1,
        )
        tokens = content.split()
        # doc 1 base unit is "a": a, aa, aaa, then leaves off the deepest prefix
        assert tokens[:3] == ["a", "aa", "aaa"]
        assert tokens[3:] == ["aaaa", "aaab"]

    def test_base_unit_differs_per_doc(self):
        t = {"type": "progressive_prefix", "max_depth": 2, "leaf_count": 1}
        assert _apply(t, doc_num=1).split()[0] == "a"
        assert _apply(t, doc_num=2).split()[0] == "b"


class TestRandomFromSet:
    def test_all_tokens_drawn_from_the_set(self):
        token_set = ["x", "y", "z"]
        content = _apply(
            {"type": "random_from_set", "token_set": token_set, "token_count": 20}
        )
        tokens = content.split()
        assert len(tokens) == 20
        assert set(tokens) <= set(token_set)

    def test_seeded_and_reproducible_per_doc(self):
        t = {"type": "random_from_set", "token_set": list("abcde"), "token_count": 30}
        assert _apply(t, doc_num=4) == _apply(t, doc_num=4)


class TestStemmableWordsTransform:
    def test_returns_empty_and_warns_only_once(self, caplog):
        # apply_transforms runs per document; the direct-use warning must not be
        # emitted once per doc.
        setup_datasets._stemmable_direct_warned = False
        t = {"type": "stemmable_words", "token_count": 100}
        with caplog.at_level("WARNING"):
            first = _apply(t, doc_num=1)
            second = _apply(t, doc_num=2)
        assert first == "" and second == ""
        warnings = [r for r in caplog.records if "stemmable_words" in r.message]
        assert len(warnings) == 1


# ---- stemmable dataset generation ------------------------------------------


class TestExtractStemmableWords:
    def test_keeps_only_stemmable_words_sorted(self, tmp_path: Path):
        # Minimal MediaWiki-style dump. "running/jumps/cats" stem to a different
        # root; "the" is a stop-length non-stemmable word.
        wiki = tmp_path / "wiki.xml"
        wiki.write_text(
            "<mediawiki><page><title>t</title><revision>"
            "<text>running jumps cats the</text>"
            "</revision></page></mediawiki>",
            encoding="utf-8",
        )
        words = setup_datasets.extract_stemmable_words_from_wiki(wiki)
        assert set(words) == {"running", "jumps", "cats"}
        # Sorted output is required for reproducible seeded sampling downstream.
        assert words == sorted(words)


class TestGenerateStemmableDataset:
    def test_reproducible_and_uses_only_extracted_words(
        self, tmp_path: Path, monkeypatch
    ):
        vocab = ["cats", "jumps", "running", "walked", "faster"]
        monkeypatch.setattr(
            setup_datasets, "extract_stemmable_words_from_wiki", lambda _f: vocab
        )
        config = {"doc_count": 3, "fields": [{"transforms": [{"token_count": 4}]}]}

        setup_datasets.generate_stemmable_dataset(
            tmp_path, tmp_path / "dummy.xml", config, "stem_a.csv"
        )
        setup_datasets.generate_stemmable_dataset(
            tmp_path, tmp_path / "dummy.xml", config, "stem_b.csv"
        )

        rows_a = list(csv.reader((tmp_path / "stem_a.csv").open()))
        rows_b = list(csv.reader((tmp_path / "stem_b.csv").open()))
        assert rows_a[0] == ["field1"]
        assert len(rows_a) == 1 + 3  # header + doc_count rows
        # Every emitted token comes from the extracted vocabulary.
        for row in rows_a[1:]:
            tokens = row[0].split()
            assert len(tokens) == 4
            assert set(tokens) <= set(vocab)
        # Seeded per-doc RNG => byte-identical output across runs.
        assert rows_a == rows_b
