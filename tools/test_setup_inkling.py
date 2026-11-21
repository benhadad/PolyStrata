"""Tests for setup.py's Inkling 2 Mini (docs/models/inkling.md): its three pinned files, the estimates for an 8 GB
card, the install by GLM's steps with no draft model and no image encoder, the model's sampling defaults in the run
config, and NVIDIA only.  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_inkling
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling"]
# setup's order: the size taken with the RAM for it, then Unsloth's 2-bit file (the least RAM), then 8-bit
SIZES = [("IQ4_NL", 17627940768), ("UD-Q2_K_XL", 9138715328), ("Q8_0", 27820099264)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling")), [m for m, _ in SIZES])
        self.assertEqual(FAM["hf"], "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/"
                                    "db61c5b492a6a034d986328d28274af0a439aa6c/")
        self.assertEqual(list(FAM["sha256"]), [f"Inkling-2-Mini-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
            if size == "UD-Q2_K_XL":   # Unsloth's repository, in the architecture's native form (inkling2)
                self.assertEqual(d["hf"], "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/"
                                          "ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/")
            else:
                self.assertNotIn("hf", d)
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling"], ("inkling2_loop", "inkling2"))
        self.assertFalse(setup.poly_batch(FAM, FAM["models"]["IQ4_NL"], 31.8, 32768))   # the engine's prompt batch

    def test_the_about_line_and_what_is_not_claimed(self):
        n = setup.poly_ram_need(FAM["models"]["IQ4_NL"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card (16.5 GiB measured for the engine)", FAM["about"])
        self.assertIs(FAM["experimental"], True)
        self.assertNotIn("unmeasured", FAM)          # measured on an RTX 2070 Super 8 GB (the comment above FAMILIES)
        self.assertIs(FAM["windows"], False)
        self.assertIs(FAM["vision"], False)
        self.assertEqual(FAM["ran_on_gb"], 8)          # run on an RTX 2070 Super 8 GB

    def test_nvidia_only(self):
        self.assertIn("inkling", setup.menu_families(False))
        self.assertNotIn("inkling", setup.menu_families(True))
        self.assertIn("inkling", setup.POLY_KV["q8_0"])      # fp16 or q8_0 caches
        sys.path.insert(0, str(ROOT / "metal"))
        import setup_mac
        self.assertNotIn("inkling", setup_mac.MAC_POLY_FAMILIES)


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling2-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["models"][size].get("hf", FAM["hf"]) + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "inkling2-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip", "--prompt-batch", "--kv"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["model_name"], f"inkling-2-mini-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Inkling 2 Mini {size}")
                self.assertNotIn("vision", cfg)
                self.assertNotIn("tool_call_recovery", cfg)
                self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_k": 50, "top_p": 1.0})
                self.assertNotIn("EXPERIMENTAL and new", out)   # measured: no "not measured yet" warning

    def test_a_sampling_block_of_the_users_is_kept(self):
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        path = self.t / "strata-inkling2-iq4_nl.json"
        path.write_text(json.dumps({**cfg, "sampling": {"temperature": 0.6}}))
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(path.read_text())["sampling"], {"temperature": 0.6})

    def test_the_size_taken_without_a_name(self):
        for ram in (64.0, 20.0):                      # IQ4_NL wants the least RAM too
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="inkling", ram=ram, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / "strata-inkling2-iq4_nl.json").is_file(), out)

    def test_windows_is_asked(self):
        code, out, _ = self.main(win=True, family="inkling", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)                # --yes takes the risk
        self.assertIn("Inkling 2 Mini has not been built or run on Windows", out)
        self.assertNotIn("has been run on Linux only", out)

    def test_one_card_and_one_request_at_a_time(self):
        # its engine code has no --layer-split and no --batch yet: setup writes neither
        code, out, _ = self.main(["--gpus", "0,1"], family="inkling", ram=64.0, n_gpus=2)
        self.assertNotEqual(code, 0)
        self.assertIn("runs on one card for now", out)
        code, out, cfg = self.main(["--gpu", "1"], family="inkling", ram=64.0, n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 1)
        code, out, cfg = self.main(["--parallel", "3"], family="inkling", ram=64.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["parallel"], 1)          # 1 also over an earlier config's "parallel"
        self.assertIn("answers one request at a time for now", out)
        self.assertNotIn("Several requests at once", out)
        code, out, cfg = self.main(["--kv", "q8_0"], family="inkling", ram=64.0)
        self.assertEqual(code, 0, out)
        self.assertIn("--kv", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "q8_0")

    def test_vision_is_not_used(self):
        code, out, cfg = self.main(["--vision", "yes"], family="inkling", ram=64.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with Inkling 2 Mini", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()
