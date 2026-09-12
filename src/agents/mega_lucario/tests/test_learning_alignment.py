"""Terminal distillation, weighted collection and selection calibration contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from agents.mega_lucario.gbdt.scripts.calibrate import fit_threshold
from agents.mega_lucario.ppo.scripts.collect import allocate_games
from agents.mega_lucario.transformer.model import ObservationTransformer
from agents.mega_lucario.transformer.train import decision_loss, losses, value_targets


class LearningAlignment(unittest.TestCase):
    """Validate learning alignment."""

    def test_weighted_allocation_is_exact_and_seat_balanced(self) -> None:
        """Check weighted allocation is exact and seat balanced."""
        weights = {"rule": 1.0, "fixed": 3.0, "off": 0.0}
        result = allocate_games(list(weights), weights, 5000)
        self.assertEqual(result, {"rule": 1250, "fixed": 3750, "off": 0})
        rounded = allocate_games(["a", "b", "c"], dict(a=1.0, b=1.0, c=1.0), 10)
        self.assertEqual(sum(rounded.values()), 10)
        self.assertTrue(all(v % 2 == 0 for v in rounded.values()))
        for bad in (dict(weights, off=-1.0), dict(weights, off=float("nan"))):
            with self.assertRaises(ValueError):
                allocate_games(list(bad), bad, 20)

    def test_value_outcomes_and_joint_selection_receive_gradients(self) -> None:
        """Check value outcomes and joint selection receive gradients."""
        torch.set_num_threads(1)
        torch.manual_seed(2)
        model = ObservationTransformer()
        x = torch.zeros(3, 22, 198)
        x[:, :, 0] = 1
        x[:, :, 4:6] = -1
        x[:, 1:13, 1] = 1
        x[:, 13:19, 1] = 2
        x[:, 19:21, 1] = 3
        x[:, 21, 1] = 4
        x[:, 19:, 2] = 1
        x[1, 19, 3] = 1
        x[1, 19, 2] = 0
        mask = torch.ones(3, 22, dtype=torch.bool)
        teacher = torch.zeros(3, 22)
        target = torch.tensor([19, 21, 20])
        owners = torch.tensor([0, 0, 1])
        first = torch.tensor([0, 2])
        outcome = torch.tensor([1.0, -1.0])
        loss, metrics, _ = decision_loss(
            model, (x, mask, teacher, target, owners, first, outcome), 1.0, 0.1, 0.25
        )
        loss.backward()
        self.assertGreater(float(model.value2.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.policy_query.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )
        np.testing.assert_array_equal(
            value_targets(torch.tensor([-1.0, 0.0, 1.0])).argmax(-1), [0, 25, 50]
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 980916)

    def test_temperature_only_softens_teacher(self) -> None:
        """Check temperature only softens teacher."""
        teacher = torch.tensor([[0.0, 4.0, -float("inf")]])
        # At temperature 2 the teacher logits are [0, 2]. Matching policy
        # logits must have zero KL without applying temperature a second time.
        policy = torch.tensor([[0.0, 2.0, -float("inf")]], requires_grad=True)
        loss, kl = losses(
            policy, teacher, torch.tensor([[True, True, False]]), torch.tensor([1]), 2.0, 0.0
        )
        self.assertLess(abs(float(kl.detach())), 1e-7)
        loss.backward()
        self.assertTrue(torch.isfinite(policy.grad).all())

    def test_calibrated_stop_ties_continue(self) -> None:
        """Check calibrated stop ties continue."""
        threshold, accuracy = fit_threshold(
            [(-2.0, True), (-1.0, True), (1.0, False), (2.0, False)]
        )
        self.assertEqual(accuracy, 1.0)
        self.assertTrue(-1.0 < threshold <= 1.0)


if __name__ == "__main__":
    unittest.main()
