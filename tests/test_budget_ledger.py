import json
import os
import shutil
import tempfile
import multiprocessing
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from budget_ledger import BudgetExhausted, BudgetLedger, CorruptLedger, cost_usd


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "budget.json")

    def tearDown(self):
        self.dir.cleanup()

    def ledger(self, cap=2.00):
        return BudgetLedger(path=self.path, cap_usd=cap)

    def test_reserve_settle_charges_actual_not_reserve(self):
        l = self.ledger()
        rid = l.reserve("test", est_input_tokens=2000, max_output_tokens=150)
        # worst case reserve: input at cache-WRITE rate + max output
        self.assertAlmostEqual(l.committed_usd(), 0.00025 + 0.000075, places=9)
        actual = l.settle(rid, {"input_tokens": 1800, "output_tokens": 40,
                                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
        self.assertAlmostEqual(actual, (1800 * 0.10 + 40 * 0.50) / 1e6, places=9)
        self.assertAlmostEqual(l._state["spent_usd"], actual, places=9)
        self.assertAlmostEqual(l.committed_usd(), actual, places=9)

    def test_cache_tokens_priced_separately(self):
        l = self.ledger()
        rid = l.reserve("test", 100, 10)
        actual = l.settle(rid, {"input_tokens": 200, "output_tokens": 40,
                                "cache_creation_input_tokens": 1600, "cache_read_input_tokens": 8000})
        expected = (200 * 0.10 + 40 * 0.50 + 1600 * 0.125 + 8000 * 0.01) / 1e6
        self.assertAlmostEqual(actual, expected, places=9)

    def test_fails_closed_at_cap(self):
        l = self.ledger(cap=0.001)
        l.reserve("a", est_input_tokens=2000, max_output_tokens=150)
        l.settle  # unused
        with self.assertRaises(BudgetExhausted):
            l.reserve("b", est_input_tokens=9000, max_output_tokens=150)

    def test_failed_call_charges_full_reserve(self):
        l = self.ledger()
        rid = l.reserve("test", est_input_tokens=2000, max_output_tokens=150)
        expected = 0.00025 + 0.000075
        charged = l.fail(rid, "http_500")
        self.assertAlmostEqual(charged, expected, places=9)
        self.assertAlmostEqual(l._state["spent_usd"], expected, places=9)

    def test_state_persists_across_instances(self):
        l = self.ledger()
        rid = l.reserve("test", 1000, 100)
        l.settle(rid, {"input_tokens": 1000, "output_tokens": 50})
        l2 = self.ledger()
        self.assertAlmostEqual(l2._state["spent_usd"], l._state["spent_usd"], places=9)

    def test_ledger_never_contains_prompt_text(self):
        l = self.ledger()
        rid = l.reserve("val-baseline", 1000, 100)
        l.settle(rid, {"input_tokens": 900, "output_tokens": 30})
        l.fail(l.reserve("test", 1000, 100), "TimeoutError")
        with open(self.path) as f:
            raw = f.read()
        self.assertNotIn('"prompt"', raw)
        for event in l._state["events"]:
            self.assertNotIn("prompt", event)

    def test_cost_usd_matches_pinned_rates(self):
        # 1M tokens of each kind at the pinned Haiku 5.5 rates.
        self.assertAlmostEqual(cost_usd(input_tokens=1_000_000), 0.10, places=9)
        self.assertAlmostEqual(cost_usd(output_tokens=1_000_000), 0.50, places=9)
        self.assertAlmostEqual(cost_usd(cache_write_tokens=1_000_000), 0.125, places=9)
        self.assertAlmostEqual(cost_usd(cache_read_tokens=1_000_000), 0.01, places=9)


if __name__ == "__main__":
    unittest.main()


class TestSafeguards(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "budget.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_reserve_uses_cache_write_rate_as_input_worst_case(self):
        # Input worst case is the 5m cache-write rate, which exceeds plain
        # input: a cache-eligible call must never cost more than reserved.
        ledger = BudgetLedger(self.path, cap_usd=2.0)
        rid = ledger.reserve("t", est_input_tokens=1000, max_output_tokens=200)
        reserved = ledger._state["reserved"][rid]["reserve_usd"]
        expected = (1000 * 0.125 + 200 * 0.50) / 1_000_000
        self.assertAlmostEqual(reserved, expected, places=9)

    def test_corrupt_ledger_fails_closed(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        with self.assertRaises(CorruptLedger):
            BudgetLedger(self.path, cap_usd=2.0)

    def test_missing_main_file_restores_from_bak(self):
        ledger = BudgetLedger(self.path, cap_usd=2.0)
        rid = ledger.reserve("t", 1000, 200)
        ledger.fail(rid, "boom")
        spent = ledger._state["spent_usd"]
        self.assertGreater(spent, 0)
        os.remove(self.path)  # lose the main file, keep the mirror
        restored = BudgetLedger(self.path, cap_usd=2.0)
        self.assertAlmostEqual(restored._state["spent_usd"], spent, places=9)

    def test_settle_without_usage_charges_full_reservation(self):
        ledger = BudgetLedger(self.path, cap_usd=2.0)
        rid = ledger.reserve("t", 1000, 200)
        reserved = ledger._state["reserved"][rid]["reserve_usd"]
        ledger.settle(rid, {})  # provider response with no usage object
        self.assertAlmostEqual(ledger._state["spent_usd"], reserved, places=9)

    def test_concurrent_reservations_cannot_overspend(self):
        # Two processes racing for the last headroom: exactly one wins.
        results = multiprocessing.Queue()

        def race():
            try:
                BudgetLedger(self.path, cap_usd=0.0003).reserve("t", 1000, 200)
                results.put("ok")
            except BudgetExhausted:
                results.put("refused")
            except Exception as e:  # pragma: no cover - diagnostic
                results.put("error:%s" % type(e).__name__)

        procs = [multiprocessing.Process(target=race) for _ in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        outcomes = sorted(results.get() for _ in procs)
        self.assertEqual(outcomes, ["ok", "refused"])
