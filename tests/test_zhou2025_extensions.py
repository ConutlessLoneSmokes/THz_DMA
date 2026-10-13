"""Small runner checks for traditional references and the physical migration."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_zhou2025_repro as runner
from _run_support import RunArchive, read_toml
from zhou2025_repro.data import ChannelData
from zhou2025_repro.mimo import parametric_mimo
from zhou2025_repro.wideband import DigitalArrayOperator


class ExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_physical_noise_and_carrier_geometry(self):
        data = SimpleNamespace(scenario="mimo", shape=(4, 8), scale=.02)
        first = runner.observation_variance(data, {}, 0., 10., .002) * data.scale**2
        data.scale = .002
        second = runner.observation_variance(data, {}, 0., 10., .002) * data.scale**2
        self.assertAlmostEqual(first, second)
        array = SimpleNamespace(scenario="dma", shape=(4, 2, 3), scale=.01)
        self.assertEqual(runner.observation_variance(array, {"snr_reference":"training_element"}, 99., 0.), 1.)
        config = read_toml(ROOT/"configs/zhou2025_carrier_300ghz_noabs.toml")
        config["mimo"].update(receive_antennas=4, transmit_antennas=8)
        config["channel"]["num_paths"] = 1
        high, high_paths = parametric_mimo(config, np.random.default_rng(1), None)
        low_config = copy.deepcopy(config)
        low_config["system"]["carrier_frequency_hz"] = 4e10
        low, low_paths = parametric_mimo(low_config, np.random.default_rng(1), None)
        self.assertEqual(high_paths, low_paths)
        self.assertAlmostEqual(float(np.mean(abs(low)**2)/np.mean(abs(high)**2)), 7.5**2, places=4)
        high_relative = high/high[0, 0]
        low_relative = low/low[0, 0]
        self.assertGreater(float(np.linalg.norm(high_relative-low_relative)), 1.)

    def test_digital_smoke_retains_no_conditional_references(self):
        config = runner.settings_from_args(SimpleNamespace(
            config=ROOT/"configs/zhou2025_wideband_digital.toml", smoke=True, seed=None))
        self.assertEqual(config["evaluation_run"]["conditional_references"], [])
        operators, platform = runner.build_operators(config, SimpleNamespace(scenario="dma", shape=(128,8,160)))
        self.assertIsNone(platform)
        self.assertTrue(all(isinstance(operator, DigitalArrayOperator) for _, operator, _ in operators))
        self.assertEqual([operator.observation_shape for _,operator,_ in operators],
                         [(1,128,8,160), (8,16,8,160)])

    def test_changed_absorption_table_rejects_existing_cache(self):
        config = runner.settings_from_args(SimpleNamespace(
            config=ROOT/"configs/zhou2025_wideband_digital.toml", smoke=True, seed=None))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"runs"/config["data"]["source_run_id"]
            source.mkdir(parents=True)
            (source/"status.json").write_text('{"status":"complete"}',encoding="utf-8")
            (source/"dataset.json").write_text('{}',encoding="utf-8")
            (source/"config_resolved.json").write_text(json.dumps({
                "config":config,"absorption_table_sha256":"changed"}),encoding="utf-8")
            table_path = root/config["atmosphere"]["table_path"]
            table_path.parent.mkdir(parents=True)
            table_path.write_text("0 0\n1 0\n",encoding="utf-8")
            with patch.object(runner,"PROJECT_ROOT",root):
                with self.assertRaisesRegex(ValueError,"original DMA cache absorption table changed"):
                    runner.prepare(self.archive(root,"data"),config)

    def write_data(self, root, scenario, channels, records=None):
        source = root/"runs/reference"
        source.mkdir(parents=True)
        metadata = {"scenario":scenario, "complex_scale":float(np.sqrt(np.mean(abs(channels["train"])**2))),
            "channel_shape":channels["train"].shape[1:], "counts":{s:len(v) for s,v in channels.items()},
            "storage":"npy", "files":{s:f"{s}.npy" for s in channels}, "kind":"clean", "physics":{},
            "split_seeds":{"train":11,"validation":12,"test":13}, "file_sha256":{}}
        (source/"dataset.json").write_text("{}", encoding="utf-8")
        for split, values in channels.items():
            np.save(source/f"{split}.npy", values.astype(np.complex64))
            if records is not None:
                (source/"parameters").mkdir(exist_ok=True)
                name = f"parameters/{split}_000000.json"
                (source/name).write_text(json.dumps(records[split]), encoding="utf-8")
                metadata["file_sha256"][name.replace("/", "\\")] = "checked by the loader outside this fixture"
        return ChannelData(source, metadata)

    def archive(self, root, name):
        archive = RunArchive(name, root/"runs"/name)
        (archive.directory/"checkpoint").mkdir(parents=True)
        return archive

    def test_mimo_parameters_and_all_reference_rows(self):
        config = read_toml(ROOT/"configs/zhou2025_mimo_extended.toml")
        config["device"] = "cpu"
        config["mimo"].update(receive_antennas=4, transmit_antennas=8)
        config["channel"]["include_absorption"] = False
        config["evaluation_run"].update(scene_limit=2, validation_scenes=2, baseline_count=3,
            pilot_counts=[8,4], snr_db=[0.,20.], lasso_steps=30, lasso_factors=[.1])
        config["job"] = {"data_run_id":"reference", "train_run_id":[], "split":"validation", "smoke":True}
        channels, records = {}, {}
        for split, seed in (("train",11),("validation",12),("test",13)):
            channels[split], records[split] = [], []
            for index in range(3):
                channel, paths = parametric_mimo(config, np.random.default_rng(np.random.SeedSequence([seed,index])), None)
                channels[split].append(channel)
                records[split].append({"scene_id":f"{seed}:{index}", "paths":paths})
            channels[split] = np.stack(channels[split])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.write_data(root, "mimo", channels, records)
            self.assertEqual(data.record("validation",1), records["validation"][1])
            archive = self.archive(root,"validation")
            with patch.object(runner,"PROJECT_ROOT",root), patch.object(runner,"load_data",return_value=data), \
                 patch.object(runner.AbsorptionTable,"from_txt",return_value=None):
                runner.evaluate(archive,config)
            result = runner.read_json(archive.directory/"metrics_summary.json")
            self.assertEqual(result["rows"],40)
            rows = [row for path in (archive.directory/"checkpoint").glob("eval_*.json") for row in runner.read_json(path)]
            oracle = [row for row in rows if row["estimator"]=="oracle_geometry_ls"]
            self.assertEqual(len(oracle),8)
            self.assertTrue(all(row["information_condition"]=="true_path_geometry_genie_unknown_gains" for row in oracle))
            self.assertTrue(all(row["nmse_linear"]>0 for row in oracle))
            data.close()

    def test_digital_lmmse_frozen_test_and_resource_metadata(self):
        rng = np.random.default_rng(19)
        channels = {split:rng.normal(size=(3,128,2,3))+1j*rng.normal(size=(3,128,2,3))
                    for split in ("train","validation","test")}
        indices = np.arange(0,128,8)
        operator = DigitalArrayOperator(indices,(128,2,3),repetitions=2)
        schedule = SimpleNamespace(pilot_indices=indices, pilot_resource_elements=32,
                                   training_symbols=2, num_switches=1)
        config = {"scenario":"dma", "device":"cpu", "data":{"batch_size":2},
            "evaluation":{"channel_nmse_subcarrier_count":2}, "evaluation_run":{
                "frontend":"digital", "conditional_references":[], "snr_reference":"training_element",
                "scene_limit":2,"validation_scenes":2,"baseline_count":3,"snr_db":[0.,20.],
                "noise_seed":9,"bootstrap_seed":8}, "job":{
                "data_run_id":"reference","train_run_id":[],"split":"validation","smoke":True}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.write_data(root,"dma",channels)
            archive = self.archive(root,"validation")
            with patch.object(runner,"PROJECT_ROOT",root), patch.object(runner,"load_data",return_value=data), \
                 patch.object(runner,"build_operators",return_value=([("digital",operator,schedule)],None)):
                runner.evaluate(archive,config)
                archive.write_json("config_resolved.json",{"config":config})
                archive.write_status("complete")
                config["job"].update(split="test",tuning_run_id="validation")
                test_archive = self.archive(root,"test")
                runner.evaluate(test_archive,config)
            result = runner.read_json(test_archive.directory/"metrics_summary.json")
            self.assertEqual(result["rows"],4)
            rows = runner.read_json(test_archive.directory/"checkpoint/eval_0_0_lmmse_reference.json")
            self.assertTrue(all(row["receiver_rf_outputs"]==6 and row["num_switches"]==0 for row in rows))
            self.assertTrue(all(row["pilot_resource_elements"]==32 and row["training_symbols"]==2 for row in rows))
            self.assertTrue(all("net_rate_bps_hz" not in row for row in rows))
            self.assertTrue(all(row["actual_rf_output_snr_db"]==row["actual_whitened_snr_db"] for row in rows))
            self.assertEqual(runner.read_json(test_archive.directory/"tuning.json"),{})
            data.close()


if __name__ == "__main__":
    unittest.main()
