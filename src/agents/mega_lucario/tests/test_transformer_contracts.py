"""Padding/ordering invariants, numerical C++ parity and incompatible asset rejection."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
from agents.mega_lucario.gbdt.generate_schema import generate
from agents.mega_lucario.transformer.features import CONTRACT, CONTRACT_ID, DECK_COUNTS, NAMES
from agents.mega_lucario.transformer.model import (
    ARCHITECTURE,
    ObservationTransformer,
    select_logits,
)
from agents.mega_lucario.transformer.scripts.export import export
from agents.mega_lucario.transformer.train import losses


class TransformerContracts(unittest.TestCase):
    """Validate transformer contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build temporary native probes and deterministic model fixtures."""
        torch.set_num_threads(1)
        torch.manual_seed(204)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.model = ObservationTransformer(len(NAMES)).eval()
        torch.save(
            dict(
                format="ptcg-transformer-checkpoint-v1",
                contract_id=CONTRACT_ID,
                contract=CONTRACT,
                architecture=ARCHITECTURE,
                deck_counts={str(k): v for k, v in DECK_COUNTS.items()},
                feature_names=NAMES,
                state_dict=cls.model.state_dict(),
            ),
            cls.root / "model.pt",
        )
        export(cls.root / "model.pt", cls.root / "model.bin")
        generate(cls.root / "mega_lucario_schema.h")
        cls.probe = cls.root / "probe"
        subprocess.run(
            shlex.split(os.environ.get("CXX", "g++"))
            + [
                "-std=c++20",
                "-O2",
                "-I",
                str(cls.root),
                str(ROOT / "src/agents/mega_lucario/transformer/cpp/probe.cpp"),
                "-o",
                str(cls.probe),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary model and probe directory."""
        cls.tmp.cleanup()

    def test_near_ties_use_original_candidate_order(self) -> None:
        """Check near ties use original candidate order."""
        logits = torch.tensor(
            [[1.0, 1.0 + 1e-6, 0.0, -float("inf")], [0.0, 1.0, 1.0 + 1e-3, -float("inf")]],
            dtype=torch.float64,
        )
        self.assertEqual(select_logits(logits).tolist(), [0, 2])

    @staticmethod
    def fixture(n: int = 120) -> torch.Tensor:
        """Create deterministic public tokens for ordering and inference tests."""
        rng = np.random.default_rng(9)
        x = np.zeros((1, n, len(NAMES)), np.float32)
        x[:, :, 0] = 1
        x[:, :, 4:6] = -1
        x[:, 1:13, 1] = 1
        x[:, 13:19, 1] = 2
        x[:, 19:-1, 1] = 3
        x[:, -1, 1] = 4
        x[:, 19:, 2] = 1
        x[:, :, 6:102] = rng.normal(size=(1, n, 96)) * 0.1
        return torch.from_numpy(x)

    def test_padding_and_permutation(self) -> None:
        """Check padding and permutation."""
        x = self.fixture(25)
        mask = torch.ones(1, 25, dtype=torch.bool)
        padded_x = torch.cat([x, torch.zeros(1, 3, len(NAMES))], 1)
        with torch.no_grad():
            expected = self.model(x, mask)
            padded = self.model(padded_x, torch.tensor([[True] * 25 + [False] * 3]))
            order = torch.tensor([*range(19), 22, 19, 21, 20, 23, 24])
            permuted = self.model(x[:, order], mask)
        torch.testing.assert_close(expected, padded[:, :25], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(expected[:, order], permuted, atol=2e-6, rtol=1e-5)
        self.assertTrue(torch.isneginf(padded[:, 25:]).all())
        with self.assertRaises(ValueError):
            self.model(
                torch.cat([self.fixture(), x[:, :1]], 1), torch.ones(1, 121, dtype=torch.bool)
            )

    def test_selected_mask_stop_and_distribution(self) -> None:
        """Check selected mask stop and distribution."""
        x = self.fixture()
        x[:, 19, 2] = 0
        x[:, 19, 3] = 1
        x[:, -1, 2] = 0
        logits, value, distribution = self.model.forward_details(
            x, torch.ones(1, 120, dtype=torch.bool)
        )
        self.assertTrue(torch.isneginf(logits[:, 19]).all())
        self.assertTrue(torch.isneginf(logits[:, -1]).all())
        self.assertEqual(distribution.shape, (1, 51))
        self.assertLess(abs(float(value.detach())), 1e-6)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 980916)

    def test_public_tokens_belief_and_overflow(self) -> None:
        """Check public tokens belief and overflow."""
        import copy

        from agents.mega_lucario.transformer.features import token_indices, tokenize, with_prefix

        ps = dict(
            hand=[],
            discard=[],
            active=[],
            bench=[],
            prize=[None] * 6,
            deckCount=54,
            handCount=0,
            asleep=False,
            paralyzed=False,
        )
        obs = dict(
            current=dict(
                yourIndex=0,
                firstPlayer=0,
                turn=1,
                supporterPlayed=False,
                energyAttached=False,
                players=[copy.deepcopy(ps), copy.deepcopy(ps)],
                stadium=[],
                looking=None,
            ),
            select=dict(
                context=0,
                minCount=1,
                maxCount=2,
                remainDamageCounter=0,
                option=[dict(type=14)] * 100,
                deck=None,
            ),
        )
        tokens = tokenize(obs, [], {})
        self.assertEqual(tokens.shape, (120, 198))
        self.assertEqual(token_indices(obs, [0, 99, -1]), [19, 118, 119])
        self.assertEqual(tokens[-1, 2], 0)
        np.testing.assert_allclose(tokens[15, 102:134].sum(), 0.9, atol=1e-6)
        np.testing.assert_allclose(tokens[16, 102:134].sum(), 0.1, atol=1e-6)
        selected = tokenize(obs, [3], {})
        np.testing.assert_array_equal(selected, with_prefix(tokens, obs, [3]))
        self.assertEqual(selected[22, 3], 1)
        self.assertEqual(selected[22, 2], 0)
        self.assertEqual(selected[-1, 2], 1)
        changed = copy.deepcopy(obs)
        changed["current"]["players"][1]["hand"] = [dict(id=999999)]
        changed["current"]["players"][1]["prize"] = [dict(id=888888)] * 6
        np.testing.assert_array_equal(tokens, tokenize(changed, [], {}))
        obs["select"]["option"].append(dict(type=14))
        with self.assertRaisesRegex(ValueError, "overflow"):
            tokenize(obs, [], {})

    def test_padded_distillation_loss_has_finite_gradients(self) -> None:
        """Check padded distillation loss has finite gradients."""
        logits = torch.tensor([[1.0, 2.0, -float("inf")]], requires_grad=True)
        teacher = torch.tensor([[2.0, 1.0, -float("inf")]])
        loss, kl = losses(
            logits, teacher, torch.tensor([[True, True, False]]), torch.tensor([0]), 1.0, 0.5
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(kl))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cpp_predictions_and_rejection(self) -> None:
        """Check cpp predictions and rejection."""
        rows = self.fixture()[0].numpy()
        text = f"120 {len(NAMES)}\n" + "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
        result = subprocess.run(
            [str(self.probe), str(self.root / "model.bin")],
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
        with torch.no_grad():
            expected = self.model.double()(
                torch.from_numpy(rows).double()[None], torch.ones(1, 120, dtype=torch.bool)
            )[0].numpy()
        np.testing.assert_allclose(
            expected, np.fromstring(result.stdout, sep=" "), atol=2e-5, rtol=2e-5
        )
        self.model.float()
        blob = (self.root / "model.bin").read_bytes()
        for changed in (
            blob[:16] + b"x" + blob[17:],
            blob[:-1],
            blob + b"extra",
            blob[:216] + b"\x00\x00\x00\x00" + blob[220:],
        ):
            path = self.root / "bad.bin"
            path.write_bytes(changed)
            self.assertNotEqual(
                subprocess.run(
                    [str(self.probe), str(path)], input=text, text=True, capture_output=True
                ).returncode,
                0,
            )

    def test_cpp_cached_encoder_multiselect_and_backends(self) -> None:
        """Check cpp cached encoder multiselect and backends."""
        rows = self.fixture()[0].numpy()
        rows[5:9, 0] = 0  # Invalid board slots must be packed out.
        rows[1:13, 4] = np.arange(12) % 17
        rows[13:19, 102:134] = 0.02
        inputs = []
        for selected in ([], [21], [21, 19], [21, 19, 22, 24, 25, 26]):
            x = rows.copy()
            x[selected, 2] = 0
            x[selected, 3] = 1
            inputs.append(x)
        stream = "".join(
            f"120 {len(NAMES)}\n" + "\n".join(" ".join(map(str, row)) for row in x) + "\n"
            for x in inputs
        )
        with torch.no_grad():
            logits, values, _ = self.model.forward_details(
                torch.from_numpy(np.stack(inputs)), torch.ones(len(inputs), 120, dtype=torch.bool)
            )
        expected = np.column_stack([logits.numpy(), values.numpy()])
        for backend in ("auto", "scalar", "reference"):
            env = dict(os.environ, PTCG_TRANSFORMER_BACKEND=backend)
            result = subprocess.run(
                [str(self.probe), str(self.root / "model.bin"), "--prepared"],
                input=stream,
                text=True,
                capture_output=True,
                check=True,
                env=env,
            )
            actual = np.array([np.fromstring(line, sep=" ") for line in result.stdout.splitlines()])
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            full = subprocess.run(
                [str(self.probe), str(self.root / "model.bin"), "--value"],
                input=stream,
                text=True,
                capture_output=True,
                check=True,
                env=env,
            )
            np.testing.assert_allclose(
                actual,
                np.array([np.fromstring(line, sep=" ") for line in full.stdout.splitlines()]),
                atol=2e-5,
                rtol=2e-5,
            )
