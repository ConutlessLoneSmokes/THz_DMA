"""Budget conservation, physical state pairing and validation-only selection."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import read_toml
from analyze_d041_budget_sweep import select_workpoint
from run_d041_budget_sweep import budget_mask, contrast_pairs, plans
from run_d041_configuration_sweep import plans as old_plans
from test_d041_diagnostic import config_for
from thz_dma.observations.pilots import full_band_pilot_indices
from zhou2025_repro.configuration_sweep import re128_mask
from zhou2025_repro.d041 import state_bank


def budget_config(budget, *, smoke=False):
    config = config_for("wide")
    config["job"]["smoke"] = smoke
    config["configuration_sweep"] = {**read_toml(ROOT/"configs/d041_budget_sweep.toml"),
        "experiment": "budget", "budget_re": budget}
    return config


class BudgetTest(unittest.TestCase):
    def test_all_budgets_and_controls_conserve_resources_and_state_prefixes(self):
        count, main_count = 0, 0
        for budget in [128, 512, 1024, 2048, 4096, 8192]:
            config = budget_config(budget)
            bank = state_bank(config)
            schedules = list(plans(config))
            by_label = {label: (states, active, meta) for label, states, active, meta in schedules}
            self.assertEqual(len(by_label), len(schedules))
            for label, states, active, meta in schedules:
                count += 1
                main_count += meta["is_main"]
                self.assertEqual(int(active.sum()), budget)
                np.testing.assert_array_equal(active.sum(1), meta["pilots_per_configuration"])
                if meta["state_bank"] == "D041_random":
                    np.testing.assert_array_equal(states, bank[meta["state_indices"]])
                if meta["state_mode"] == "repeat":
                    np.testing.assert_array_equal(states, np.repeat(states[:1], len(states), axis=0))
                if meta["state_mode"] == "interleaved":
                    np.testing.assert_array_equal(active.sum(0), budget//128)
                elif meta["state_bank"] == "D041_random":
                    np.testing.assert_array_equal(np.flatnonzero(active[0]), full_band_pilot_indices(128, active.sum(1)[0]))
            for label, control, kind in contrast_pairs([dict(label=label, **meta) for label, _, _, meta in schedules]):
                a, b = by_label[label], by_label[control]
                self.assertEqual(a[1].sum(), b[1].sum())
                self.assertEqual(len(a[0]), len(b[0]))
                if kind in ("state_diversity", "random_minus_legacy_bank"):
                    np.testing.assert_array_equal(a[1], b[1])
                else:
                    np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(main_count, 24)
        self.assertEqual(count, 78)

    def test_128_interleaving_and_legacy_controls_are_preserved(self):
        for j in [1, 2, 4, 8, 16]:
            np.testing.assert_array_equal(budget_mask(j, 128//j), re128_mask(j))
            np.testing.assert_array_equal(budget_mask(j, 128//j, interleaved=True), re128_mask(j, interleaved=True))
        config = budget_config(128)
        legacy = copy.deepcopy(config)
        legacy["configuration_sweep"].update(experiment="re128", re128_configuration_counts=[1, 2, 4, 8, 16])
        old = list(old_plans(legacy))
        new = [p for p in plans(config) if p[0].startswith("legacy_")]
        self.assertEqual(len(new), 13)
        for a, b in zip(old, new):
            self.assertEqual("legacy_"+a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_array_equal(a[2], b[2])

    def test_full_tones_skip_identical_coverage_control_but_keep_repeat(self):
        schedules = {label for label, *_ in plans(budget_config(8192))}
        self.assertIn("re8192_J64_K128_distinct", schedules)
        self.assertIn("re8192_J64_K128_repeat", schedules)
        self.assertNotIn("re8192_J64_K128_interleaved", schedules)
        self.assertIn("re8192_J128_K64_interleaved", schedules)
        with self.assertRaises(ValueError):
            budget_mask(3, 8, interleaved=True)
        with self.assertRaises(ValueError):
            budget_mask(193, 128)

    def test_references_and_smoke_counts_are_explicit(self):
        references = list(plans(budget_config(0)))
        self.assertEqual([p[0] for p in references], ["digital", "dma_J192"])
        self.assertTrue(all(not p[3]["is_main"] for p in references))
        total = sum(len(list(plans(budget_config(r, smoke=True)))) for r in [0, 128, 512, 1024, 2048, 4096, 8192])
        self.assertEqual(total, 35)

    def test_validation_workpoint_uses_gate_then_re_and_symbols(self):
        frame = pd.DataFrame([dict(operator=label, is_main=main, budget_re=re, pilot_re=re,
            training_symbols=j, configurations=j, pilots_per_configuration=128,
            recovery_scene_fraction=fraction, run_id="validation") for label, main, re, j, fraction in [
                ("legacy", False, 128, 1, 1.), ("failed", True, 128, 1, .94),
                ("many_symbols", True, 512, 8, .99), ("chosen", True, 512, 4, .95),
                ("larger", True, 1024, 2, 1.)]])
        chosen = select_workpoint(frame, smoke=False, required_fraction=.95)
        self.assertEqual(chosen["operator"], "chosen")
        self.assertIsNone(select_workpoint(frame, smoke=True, required_fraction=.95))
        self.assertIsNone(select_workpoint(frame.loc[frame.operator == "failed"], smoke=False, required_fraction=.95))


if __name__ == "__main__":
    unittest.main()
