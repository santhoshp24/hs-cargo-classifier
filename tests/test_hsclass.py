import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsclass import cli  # noqa: E402
from hsclass.generate import generate  # noqa: E402
from hsclass.hs_reference import (CATEGORY_BY_CHAPTER, CHAPTERS, VALID_CHAPTERS,  # noqa: E402
                                  category_for, is_edible)
from hsclass.lookup import Lookup  # noqa: E402
from hsclass.normalise import normalise  # noqa: E402
from hsclass.pipeline import REVIEW, classify_rows  # noqa: E402
from hsclass.providers import (HeuristicProvider, ProviderError, build_providers,  # noqa: E402
                               parse_llm_response)


class FakeProvider:
    """Answers from a fixed table and records every key it was asked about."""

    def __init__(self, name, answers, fail=False):
        self.name, self.answers, self.fail, self.asked = name, answers, fail, []

    def classify(self, batch):
        if self.fail:
            raise ProviderError("boom")
        self.asked += [k for k, _ in batch]
        return {k: self.answers[k] for k, _ in batch if k in self.answers}


COCOA = normalise("cocoa beans")  # "BEANS COCOA": keys are sorted bags of words
TEA = normalise("black tea")


def rows(*descs):
    return [{"record_id": f"R{i}", "d": d} for i, d in enumerate(descs, 1)]


class ReferenceTests(unittest.TestCase):
    def test_chapter_set(self):
        self.assertEqual(len(VALID_CHAPTERS), 96)
        self.assertNotIn(77, VALID_CHAPTERS)
        self.assertEqual(set(CHAPTERS), set(CATEGORY_BY_CHAPTER))

    def test_category_and_edible(self):
        self.assertEqual(category_for(2), "Food & Beverages")
        self.assertTrue(is_edible(2))
        self.assertFalse(is_edible(31))
        self.assertEqual(category_for(77), "Unclassified")


class NormaliseTests(unittest.TestCase):
    def test_variants_share_a_key(self):
        a = normalise("1X40HC 24 PALLETS FROZEN BONELESS BEEF")
        b = normalise("boneless frozen beef - 18.5 MT")
        c = normalise("FROZEN BONELESS BEEF ORIGIN BRAZIL")
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual(a, "BEEF BONELESS FROZEN")

    def test_empty_and_noise_only(self):
        self.assertEqual(normalise(""), "")
        self.assertEqual(normalise(None), "")
        self.assertEqual(normalise("20 PALLETS 40HC 12.5 MT"), "")


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "curated.csv").write_text("description,hs_chapter\nFROZEN BEEF,2\nbad row,999\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_curated_loaded_and_invalid_skipped(self):
        lk = Lookup(self.dir / "curated.csv", self.dir / "learned.csv")
        self.assertEqual(lk.get(normalise("beef frozen")).chapter, 2)
        self.assertEqual(lk.sizes["curated"], 1)

    def test_learn_rules_and_roundtrip(self):
        lk = Lookup(self.dir / "curated.csv", self.dir / "learned.csv")
        self.assertTrue(lk.learn("COCOA BEANS", 18, "x", 0.9))
        self.assertFalse(lk.learn("COCOA BEANS", 18, "x", 0.9), "no duplicates")
        self.assertFalse(lk.learn("WHATEVER", 77, "x", 0.9), "reserved chapter refused")
        self.assertFalse(lk.learn("BEEF FROZEN", 3, "x", 0.9), "curated entries are never overridden")
        lk.save_learned()
        again = Lookup(self.dir / "curated.csv", self.dir / "learned.csv")
        hit = again.get("COCOA BEANS")
        self.assertEqual((hit.chapter, hit.tier), (18, "learned"))


class ParseTests(unittest.TestCase):
    IDS = {"1": "A", "2": "B", "3": "C"}

    def test_valid_and_fenced(self):
        txt = '```json\n{"results":[{"id":"1","chapter":2,"confidence":0.9}]}\n```'
        self.assertEqual(parse_llm_response(txt, self.IDS), {"A": (2, 0.9)})

    def test_bad_items_are_dropped(self):
        txt = ('{"results":[{"id":"1","chapter":77,"confidence":0.9},'
               '{"id":"2","chapter":98,"confidence":0.9},'
               '{"id":"3","chapter":"x","confidence":0.9},'
               '{"id":"9","chapter":2,"confidence":0.9},'
               '{"id":"2","chapter":5,"confidence":7}]}')
        self.assertEqual(parse_llm_response(txt, self.IDS), {"B": (5, 1.0)})

    def test_top_level_list_and_garbage(self):
        self.assertEqual(parse_llm_response('[{"id":"1","chapter":4,"confidence":0.5}]', self.IDS), {"A": (4, 0.5)})
        with self.assertRaises(ProviderError):
            parse_llm_response("not json", self.IDS)
        with self.assertRaises(ProviderError):
            parse_llm_response('{"foo": 1}', self.IDS)


class HeuristicTests(unittest.TestCase):
    def classify(self, text):
        return HeuristicProvider().classify([("k", text)]).get("k")

    def test_plurals_and_phrases(self):
        self.assertEqual(self.classify("5 DRUMS OF TABLE POTATOES")[0], 7)
        self.assertEqual(self.classify("WRIST WATCHES")[0], 91)
        self.assertEqual(self.classify("SPORTS SHOES")[0], 64)

    def test_unknown_and_ambiguous(self):
        self.assertIsNone(self.classify("ACETONE"))
        ch, conf = self.classify("PVC PIPES") or (None, 0)
        self.assertEqual(ch, 39)
        _, conf = self.classify("LEATHER HANDBAGS")
        self.assertLess(conf, 0.75, "ambiguous input must fall below the default threshold")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "curated.csv").write_text("description,hs_chapter\nFROZEN BEEF,2\n", encoding="utf-8")
        self.lk = Lookup(self.dir / "curated.csv", self.dir / "learned.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def run_pipeline(self, descs, providers, **kw):
        return classify_rows(rows(*descs), "d", "record_id", self.lk, providers, **kw)

    def test_lookup_first_and_dedup(self):
        p = FakeProvider("p", {COCOA: (18, 0.95)})
        res, st = self.run_pipeline(["FROZEN BEEF", "20 PALLETS cocoa beans", "COCOA BEANS", "cocoa beans 5MT"], [p])
        self.assertEqual([r.method for r in res], ["lookup_curated", "p", "p", "p"])
        self.assertEqual(p.asked, [COCOA], "one provider question per unique key, none for lookup hits")
        self.assertEqual(st["unique_descriptions"], 2)

    def test_low_confidence_passes_to_next_provider(self):
        a = FakeProvider("a", {COCOA: (18, 0.40)})
        b = FakeProvider("b", {COCOA: (18, 0.90)})
        res, _ = self.run_pipeline(["cocoa beans"], [a, b])
        self.assertEqual((res[0].method, res[0].chapter), ("b", 18))

    def test_provider_failure_falls_back(self):
        a = FakeProvider("a", {}, fail=True)
        b = FakeProvider("b", {COCOA: (18, 0.90)})
        res, st = self.run_pipeline(["cocoa beans"], [a, b])
        self.assertEqual(res[0].method, "b")
        self.assertEqual(st["provider_errors"]["a"], 1)

    def test_review_when_nothing_confident(self):
        a = FakeProvider("a", {COCOA: (18, 0.40)})
        res, st = self.run_pipeline(["cocoa beans", "20 PALLETS 40HC", "mystery cargo"], [a])
        self.assertEqual([r.method for r in res], [REVIEW] * 3)
        self.assertEqual(res[0].candidate, 18)
        self.assertIsNone(res[1].chapter)
        self.assertEqual(st["unique_in_review"], 3)

    def test_only_confident_answers_are_learned(self):
        a = FakeProvider("a", {COCOA: (18, 0.95), TEA: (9, 0.50)})
        _, st = self.run_pipeline(["cocoa beans", "black tea"], [a])
        self.assertEqual(st["learned_new"], 1)
        self.assertIsNotNone(self.lk.get(COCOA))
        self.assertIsNone(self.lk.get(TEA))
        self.assertTrue((self.dir / "learned.csv").exists())

    def test_second_run_needs_no_provider(self):
        a = FakeProvider("a", {COCOA: (18, 0.95)})
        self.run_pipeline(["cocoa beans"], [a])
        b = FakeProvider("b", {})
        res, _ = self.run_pipeline(["COCOA BEANS 20 PALLETS"], [b])
        self.assertEqual(res[0].method, "lookup_learned")
        self.assertEqual(b.asked, [])

    def test_no_learn_flag(self):
        a = FakeProvider("a", {COCOA: (18, 0.95)})
        self.run_pipeline(["cocoa beans"], [a], learn=False)
        self.assertFalse((self.dir / "learned.csv").exists())

    def test_output_row_fields(self):
        res, _ = self.run_pipeline(["frozen beef"], [])
        row = res[0].as_row()
        self.assertEqual((row["hs_chapter"], row["cargo_category"], row["edible"]), (2, "Food & Beverages", "Edible"))


class ProviderFactoryTests(unittest.TestCase):
    def test_build(self):
        self.assertEqual([p.name for p in build_providers("anthropic,groq,heuristic")], ["anthropic", "groq", "heuristic"])
        with self.assertRaises(ValueError):
            build_providers("gpt")
        with self.assertRaises(ValueError):
            build_providers(" , ")


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class LlmProviderWiringTests(unittest.TestCase):
    BATCH = [("BEANS COCOA", "cocoa beans 20 pallets")]
    GOOD = '{"results":[{"id":"1","chapter":18,"confidence":0.95}]}'

    def test_groq_retries_then_succeeds(self):
        from unittest import mock
        from hsclass import providers
        ok = FakeResponse(200, {"choices": [{"message": {"content": self.GOOD}}]})
        with mock.patch.dict("os.environ", {"GROQ_API_KEY": "k"}), \
                mock.patch("requests.post", side_effect=[FakeResponse(429), FakeResponse(503), ok]) as post, \
                mock.patch.object(providers.time, "sleep"):
            out = providers.GroqProvider().classify(self.BATCH)
        self.assertEqual(out, {"BEANS COCOA": (18, 0.95)})
        self.assertEqual(post.call_count, 3)
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["response_format"], {"type": "json_object"})

    def test_groq_gives_up_after_three_attempts(self):
        from unittest import mock
        from hsclass import providers
        with mock.patch.dict("os.environ", {"GROQ_API_KEY": "k"}), \
                mock.patch("requests.post", return_value=FakeResponse(429)), \
                mock.patch.object(providers.time, "sleep"):
            with self.assertRaises(ProviderError):
                providers.GroqProvider().classify(self.BATCH)

    def test_anthropic_wiring_with_fake_sdk(self):
        from types import ModuleType, SimpleNamespace
        from unittest import mock
        from hsclass import providers
        captured = {}

        class FakeMessages:
            def create(self, **kw):
                captured.update(kw)
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=LlmProviderWiringTests.GOOD)])

        fake = ModuleType("anthropic")
        fake.Anthropic = lambda **kw: SimpleNamespace(messages=FakeMessages())
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            out = providers.AnthropicProvider().classify(self.BATCH)
        self.assertEqual(out, {"BEANS COCOA": (18, 0.95)})
        self.assertEqual(captured["temperature"], 0)
        self.assertIn("cocoa beans 20 pallets", captured["messages"][0]["content"])
        self.assertIn("18 ", captured["messages"][0]["content"])

    def test_anthropic_errors_become_provider_errors(self):
        from types import ModuleType
        from unittest import mock
        from hsclass import providers

        def explode(**kw):
            raise ConnectionError("network down")

        fake = ModuleType("anthropic")
        fake.Anthropic = explode
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(ProviderError):
                providers.AnthropicProvider().classify(self.BATCH)


class GenerateAndCliTests(unittest.TestCase):
    def test_generation_is_deterministic_and_valid(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(Path(a), rows=200, seed=7)
            generate(Path(b), rows=200, seed=7)
            self.assertEqual((Path(a) / "sample_cargo.csv").read_text(), (Path(b) / "sample_cargo.csv").read_text())
            with open(Path(a) / "sample_cargo.csv", newline="", encoding="utf-8") as fh:
                data = list(csv.DictReader(fh))
            self.assertEqual(len(data), 200)
            self.assertTrue(all(int(r["true_hs_chapter"]) in VALID_CHAPTERS for r in data))
            self.assertFalse(any("HS " in r["cargo_description"].upper() for r in data), "no code leakage")

    def test_end_to_end_cli(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(buf):
            d, out = Path(t) / "data", Path(t) / "out"
            self.assertEqual(cli.main(["generate", "--out", str(d), "--rows", "400"]), 0)
            args = ["classify", "--input", str(d / "sample_cargo.csv"), "--curated", str(d / "curated_lookup.csv"),
                    "--learned", str(d / "learned_lookup.csv"), "--out", str(out), "--providers", "heuristic"]
            self.assertEqual(cli.main(args), 0)
            self.assertEqual(cli.main(["evaluate", "--classified", str(out / "classified.csv"),
                                       "--truth", str(d / "sample_cargo.csv")]), 0)
            # review -> learn loop
            with open(out / "review_queue.csv", encoding="utf-8") as fh:
                queue = list(csv.DictReader(fh))
            self.assertTrue(queue)
            queue[0]["hs_chapter"] = "29"
            with open(out / "reviewed.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(queue[0]))
                w.writeheader()
                w.writerows(queue)
            self.assertEqual(cli.main(["learn", "--reviewed", str(out / "reviewed.csv"),
                                       "--curated", str(d / "curated_lookup.csv")]), 0)
            self.assertEqual(Lookup(d / "curated_lookup.csv", None).get(normalise(queue[0]["description"])).chapter, 29)
            self.assertEqual(cli.main(["classify", "--input", "missing.csv"]), 1)

    def test_missing_column_is_reported(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(buf):
            p = Path(t) / "x.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            self.assertEqual(cli.main(["classify", "--input", str(p), "--out", t]), 1)
        self.assertIn("not found", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
