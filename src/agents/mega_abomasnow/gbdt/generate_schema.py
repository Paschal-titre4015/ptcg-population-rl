"""Generate the C++ contract from the shared deck/feature schema (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from agents.mega_abomasnow.gbdt.rich_features import names as rich_names
from agents.mega_abomasnow.transformer.features import CONTRACT_ID, SCORE_TIE_TOLERANCE

SCHEMA_PATH = ROOT / "src/agents/mega_abomasnow/gbdt/schema.json"


def generate(output: Path) -> Path:
    """Write the deck, feature names, and model identifiers to a C++ header."""
    raw = SCHEMA_PATH.read_bytes()
    schema = json.loads(raw)
    names = schema["feature_names"]
    if len(names) != len(set(names)) or sum(schema["deck_counts"].values()) != 60:
        raise ValueError("Invalid feature/deck schema")
    ids = sorted(map(int, schema["deck_counts"]))
    if names[schema["rich_features"]["base_width"] :] != rich_names(ids, schema["card_fields"]):
        raise ValueError("Rich GBDT feature names do not match the schema")
    lines = [
        "#pragma once",
        "#include <array>",
        "#include <string>",
        "#include <vector>",
        "namespace mega_abomasnow_schema {",
        f'inline constexpr char ID[] = "{hashlib.sha256(raw).hexdigest()}";',
        f"inline constexpr double TRANSFORMER_TIE = {SCORE_TIE_TOLERANCE};",
        f'inline constexpr char TRANSFORMER_ID[] = "{CONTRACT_ID}";',
        f"inline constexpr int DOMAIN_OFFSET = {schema['domain_offset']};",
        f'inline constexpr char VERSION[] = "{schema["version"]}";',
        f"inline constexpr std::array<int, {len(ids)}> IDS{{" + ",".join(map(str, ids)) + "};",
        f"inline constexpr std::array<int, {len(ids)}> COUNTS{{"
        + ",".join(str(schema["deck_counts"][str(i)]) for i in ids)
        + "};",
        "inline const std::vector<std::string> NAMES{",
    ]
    profile = schema["profile"]
    lines = lines[:-1]
    lines.append(f"inline constexpr int ATTACK_TYPE = {profile['attack_type']};")
    for key in ("pokemon_ids", "attackers", "basic_energy_ids"):
        values = profile[key]
        lines.append(
            f"inline const std::vector<int> {key.upper()}{{" + ",".join(map(str, values)) + "};"
        )
    lines.append(
        "inline const std::vector<std::array<int,2>> EVOLUTIONS{"
        + ",".join("{" + ",".join(map(str, pair)) + "}" for pair in profile["evolutions"])
        + "};"
    )
    lines.append("inline const std::vector<std::string> NAMES{")
    lines.extend("    " + json.dumps(name) + "," for name in names)
    lines += ["};", "}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    generate(ROOT / "build/generated/mega_abomasnow_schema.h")
