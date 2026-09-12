"""Contract tests for hidden information, selection, replay alignment and trees."""

from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
from agents.mega_lucario.gbdt.features import (
    DECK_COUNTS,
    NAMES,
    SCHEMA_ID,
    action_queries,
    check_deck,
    feature_row,
    options_for,
)
from agents.mega_lucario.gbdt.generate_schema import generate
from agents.mega_lucario.replay import inspect_replay

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def observation() -> dict[str, Any]:
    """Create a minimal public observation for contract tests."""
    card = dict(id=677, serial=1, playerIndex=0)
    player = dict(
        hand=[card],
        active=[],
        bench=[],
        discard=[],
        prize=[None] * 6,
        handCount=1,
        deckCount=53,
        asleep=False,
        paralyzed=False,
    )
    enemy = dict(player, hand=None)
    return dict(
        current=dict(
            turn=1,
            yourIndex=0,
            firstPlayer=0,
            supporterPlayed=False,
            energyAttached=False,
            stadium=[],
            looking=None,
            players=[player, enemy],
        ),
        select=dict(
            context=0,
            minCount=1,
            maxCount=1,
            remainDamageCounter=0,
            deck=None,
            effect=None,
            contextCard=None,
            option=[dict(type=7, index=0)],
        ),
        logs=[],
    )


TABLE = {
    677: dict(stage1=False, stage2=False, ex=False, megaEx=False, weakness=None, resistance=None)
}


class FeatureContracts(unittest.TestCase):
    """Validate feature contracts."""

    def test_rich_schema_and_canonical_bench(self) -> None:
        """Check rich schema and canonical bench."""
        from agents.mega_lucario.gbdt.features import IDS, SCHEMA
        from agents.mega_lucario.gbdt.rich_features import names

        self.assertEqual(NAMES[358:], names(IDS, SCHEMA["card_fields"]))
        self.assertEqual(len(NAMES), 4926)
        obs = observation()
        bench = [
            dict(id=677, hp=80, maxHp=80, energies=[6], serial=20),
            dict(id=677, hp=60, maxHp=80, energies=[6, 6], serial=21),
        ]
        obs["current"]["players"][0]["bench"] = bench
        before = feature_row(obs, 0, [], TABLE)
        bench.reverse()
        for c in bench:
            c["serial"] += 100
        self.assertEqual(before, feature_row(obs, 0, [], TABLE))
        bench[0]["energies"] = [5, 5]
        self.assertNotEqual(before[358:], feature_row(obs, 0, [], TABLE)[358:])

    def test_public_opponent_identity_and_lucario_resources(self) -> None:
        """Check public opponent identity and lucario resources."""
        obs = observation()
        table = dict(TABLE)
        for id in (6, 675, 676, 678, 999, 1000):
            table[id] = dict(TABLE[677])
        me, op = obs["current"]["players"]
        op["discard"] = [dict(id=999)]
        before = feature_row(obs, 0, [], table)
        op["discard"] = [dict(id=1000)]
        self.assertNotEqual(before[358:], feature_row(obs, 0, [], table)[358:])
        me["bench"] = [dict(id=675), dict(id=676), dict(id=677)]
        me["hand"] += [dict(id=6), dict(id=678)]
        row = feature_row(obs, 0, [], table)
        self.assertEqual(row[NAMES.index("lucario_lunar_cycle_resource_ready")], 1)
        self.assertEqual(row[NAMES.index("lucario_riolu_evolution_pairs")], 1)
        self.assertEqual(row[:358], feature_row(obs, 0, [], table, rich=False))

    def test_lucario_attack_costs_and_dusk_ball(self) -> None:
        """Check lucario attack costs and dusk ball."""
        obs = observation()
        table = {i: dict(TABLE[677]) for i in (675, 676, 678, 1102)}
        table.update(TABLE)
        me, op = obs["current"]["players"]
        me["active"] = [dict(id=676, hp=110, maxHp=110, energies=[6])]
        me["bench"] = [dict(id=678, hp=340, maxHp=340, energies=[6])]
        op["active"] = [dict(id=677, hp=60, maxHp=80)]
        obs["select"]["option"] = [dict(type=13, attackId=980)]
        row = feature_row(obs, 0, [], table)
        self.assertEqual(row[354], 0)  # No benched Lunatone: Cosmic Beam fails.
        self.assertEqual(row[NAMES.index("lucario_ready_plus_0_678")], 1)
        self.assertEqual(row[NAMES.index("lucario_mega_ready_brave")], 0)
        me["bench"].append(dict(id=675, hp=110, maxHp=110, energies=[6]))
        row = feature_row(obs, 0, [], table)
        self.assertEqual(row[354], 70)
        self.assertEqual(row[NAMES.index("lucario_ready_plus_0_675")], 0)
        obs["select"]["effect"] = dict(id=1102)
        obs["current"]["looking"] = [dict(id=678)]
        row = feature_row(obs, 0, [], table)
        self.assertEqual(row[NAMES.index("lucario_dusk_search_prompt")], 1)
        self.assertEqual(row[NAMES.index("rich_public_search_hash_67_" + str(1 + 678 % 67))], 1)

    def test_hidden_information_and_seat_symmetry(self) -> None:
        """Check hidden information and seat symmetry."""
        obs = observation()
        expected = feature_row(obs, 0, [], TABLE)
        obs["current"]["players"][1]["hand"] = [dict(id=999)]
        obs["current"]["players"][1]["deck"] = [dict(id=1000)]
        self.assertEqual(expected, feature_row(obs, 0, [], TABLE))
        obs = observation()
        obs["current"]["players"].reverse()
        obs["current"]["yourIndex"] = 1
        obs["current"]["firstPlayer"] = 1
        obs["current"]["players"][1]["hand"][0]["playerIndex"] = 1
        self.assertEqual(expected, feature_row(obs, 0, [], TABLE))

    def test_physical_position_is_not_a_feature(self) -> None:
        """Check physical position is not a feature."""
        obs = observation()
        expected = feature_row(obs, 0, [], TABLE)
        original = copy.deepcopy(obs["current"]["players"][0]["hand"][0])
        obs["current"]["players"][0]["hand"] = [None, original]
        obs["select"]["option"][0]["index"] = 1
        original["serial"] = 99
        self.assertEqual(expected, feature_row(obs, 0, [], TABLE))

    def test_stop_mask_and_remaining_options(self) -> None:
        """Check stop mask and remaining options."""
        obs = observation()
        obs["select"].update(
            minCount=1, maxCount=2, option=[dict(type=0, number=i) for i in range(3)]
        )
        self.assertEqual(options_for(obs, []), [0, 1, 2])
        self.assertEqual(options_for(obs, [1]), [0, 2, -1])
        self.assertEqual(options_for(obs, [1, 0]), [])
        queries = list(action_queries(obs, [1], TABLE))
        self.assertEqual([q["target"] for q in queries], [1, -1])
        with self.assertRaises(ValueError):
            list(action_queries(obs, [1, 1], TABLE))
        with self.assertRaises(ValueError):
            list(action_queries(obs, [], TABLE))
        with self.assertRaises(ValueError):
            check_deck([6] * 60)

    def test_replay_requires_aligned_complete_success(self) -> None:
        """Check replay requires aligned complete success."""
        header = dict(
            type="header", format="ptcg-replay-v1", game=0, agents=["mega_lucario", "mega_lucario"]
        )
        decision = dict(
            type="decision",
            game=0,
            step=0,
            seat=0,
            agent="mega_lucario",
            observation=observation(),
            action=[0],
        )
        terminal = dict(type="terminal", game=0, steps=1, result=0, agent_errors=0, engine_errors=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "game.jsonl"

            def write(records: Sequence[dict[str, Any]]) -> None:
                """Write the supplied replay records to a temporary fixture."""
                path.write_text("".join(json.dumps(r) + "\n" for r in records))

            write([header, decision, terminal])
            inspect_replay(path)
            for changed in (
                [header, decision],
                [header, dict(decision, step=1), terminal],
                [header, dict(decision, action=[1]), terminal],
                [header, decision, dict(terminal, engine_errors=1)],
            ):
                write(changed)
                with self.assertRaises(ValueError):
                    inspect_replay(path)


class NativeTreeContracts(unittest.TestCase):
    """Validate native tree contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build temporary native probes and deterministic model fixtures."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        generate(cls.root / "mega_lucario_schema.h")
        cls.binary = cls.root / "probe"
        subprocess.run(
            shlex.split(os.environ.get("CXX", "g++"))
            + [
                "-std=c++20",
                "-O0",
                "-I",
                str(cls.root),
                str(ROOT / "src/agents/mega_lucario/gbdt/cpp/probe.cpp"),
                "-o",
                str(cls.binary),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary model and probe directory."""
        cls.tmp.cleanup()

    def model_text(self) -> str:
        """Build a tiny native tree model for boundary and schema tests."""
        lines = [
            "PTCG_GBDT_V2",
            SCHEMA_ID,
            "MODEL_ID " + "a" * 64,
            f"FEATURES {len(NAMES)}",
            *NAMES,
            f"DECK {len(DECK_COUNTS)}",
            *[f"{k} {v}" for k, v in sorted(DECK_COUNTS.items())],
            f"ROUTE_FEATURE {NAMES.index('context_0')}",
            "BANKS 2",
            "BANK 0",
            "TREES 1",
            "NODES 3",
            "0 2 1 2 0",
            "-1 0 -1 -1 0.25",
            "-1 0 -1 -1 -0.5",
            "BANK 1",
            "TREES 1",
            "NODES 1",
            "-1 0 -1 -1 1.5",
            "ARCHIVE_BYTES 2",
            "{}",
            "END",
        ]
        return "\n".join(lines) + "\n"

    def run_probe(
        self, text: str, rows: Sequence[Sequence[float]]
    ) -> subprocess.CompletedProcess[str]:
        """Run the native probe with the supplied model and input rows."""
        path = self.root / "test.gbdt"
        path.write_text(text)
        data = f"{len(rows)} {len(NAMES)}\n" + "\n".join(" ".join(map(str, r)) for r in rows) + "\n"
        return subprocess.run(
            [str(self.binary), str(path)], input=data, capture_output=True, text=True
        )

    def test_numeric_boundary_and_main_routing(self) -> None:
        """Check numeric boundary and main routing."""
        rows = [[0.0] * len(NAMES) for _ in range(3)]
        rows[0][0] = 2.0
        rows[1][0] = 2.000001
        rows[2][NAMES.index("context_0")] = 1.0
        result = self.run_probe(self.model_text(), rows)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(map(float, result.stdout.split())), [0.25, -0.5, 1.5])

    def test_wrong_schema_deck_and_malformed_tree_fail(self) -> None:
        """Check wrong schema deck and malformed tree fail."""
        text = self.model_text()
        rows = [[0.0] * len(NAMES)]
        for changed in (
            text.replace("PTCG_GBDT_V2", "PTCG_GBDT_V1"),
            text.replace(SCHEMA_ID, "wrong"),
            text.replace("6 13\n", "6 12\n"),
            text.replace("0 2 1 2 0", "0 2 0 2 0"),
            text.replace("0 2 1 2 0", "0 nan 1 2 0"),
            text + "extra",
        ):
            self.assertNotEqual(self.run_probe(changed, rows).returncode, 0)

    def test_v2_archive_length_and_truncation(self) -> None:
        """Check v2 archive length and truncation."""
        text = self.model_text()
        rows = [[0.0] * len(NAMES)]
        self.assertEqual(self.run_probe(text, rows).returncode, 0)
        for changed in (
            text.replace("BYTES 2", "BYTES 3"),
            text.replace("BYTES 2", "BYTES -1"),
            text.replace("BYTES 2", "BYTES 67108865"),
            text[:-6],
        ):
            self.assertNotEqual(self.run_probe(changed, rows).returncode, 0)


class TrainingContracts(unittest.TestCase):
    """Validate training contracts."""

    def test_context_weights_count_decisions_once(self) -> None:
        """Check context weights count decisions once."""
        from agents.mega_lucario.gbdt.train import context_weights

        meta = [dict(game="g", step=i, seat=0, context=c) for i, c in enumerate([0, 1, 14])]
        meta.append(dict(meta[2]))
        self.assertEqual(
            context_weights(meta, [2, 2, 2, 2]).tolist(),
            [1.0, 1.0, 1.0, 1.0] + [(1 / 6) * (6**0.5)] * 4,
        )

    def test_bundle_roundtrip_and_checksum(self) -> None:
        """Check bundle roundtrip and checksum."""
        import lightgbm as lgb
        import numpy as np

        from agents.mega_lucario.gbdt.bundle import load_bundle
        from agents.mega_lucario.gbdt.train import export_model

        x = np.zeros((4, len(NAMES)))
        x[:, 0] = [0, 1, 0, 1]
        b = lgb.train(
            dict(
                objective="lambdarank",
                verbosity=-1,
                num_threads=1,
                min_data_in_leaf=1,
                min_data_in_bin=1,
            ),
            lgb.Dataset(x, label=[0, 1, 0, 1], group=[2, 2], feature_name=NAMES),
            num_boost_round=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gbdt"
            identity = export_model(
                dict(main=b, general=b),
                path,
                SCHEMA_ID,
                NAMES,
                DECK_COUNTS,
                dict(schema_id=SCHEMA_ID),
            )
            report, boosters = load_bundle(path)
            self.assertEqual(report["model_id"], identity)
            np.testing.assert_allclose(
                b.predict(x), boosters["main"].predict(x), atol=1e-10, rtol=1e-10
            )
            path.write_bytes(path.read_bytes().replace(b"ARCHIVE_BYTES", b"ARCHIVE_BYTEX"))
            with self.assertRaises(ValueError):
                load_bundle(path)
