"""Read complete, successful arena replays; reject unusable training evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


def sha256(path: Path | str) -> str:
    """Return the SHA-256 digest of a replay or model file."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_replay(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a complete, successful replay and return its header and terminal record."""
    header = terminal = None
    decisions = 0
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            kind = record.get("type")
            if header is None:
                if kind != "header" or record.get("format") != "ptcg-replay-v1":
                    raise ValueError(f"Missing replay header: {path}")
                header = record
            elif terminal is not None:
                raise ValueError(f"Data after terminal record: {path}")
            elif kind == "decision":
                if record["game"] != header["game"] or record["step"] != decisions:
                    raise ValueError(f"Non-contiguous decisions: {path}")
                seat = record["seat"]
                if seat not in (0, 1) or record["agent"] != header["agents"][seat]:
                    raise ValueError(f"Invalid replay seat/agent: {path}")
                if record["observation"]["current"]["yourIndex"] != seat:
                    raise ValueError(f"Observation belongs to another player: {path}")
                selection, action = record["observation"]["select"], record["action"]
                if not (
                    selection["minCount"] <= len(action) <= selection["maxCount"]
                    and len(set(action)) == len(action)
                    and all(type(i) is int and 0 <= i < len(selection["option"]) for i in action)
                ):
                    raise ValueError(f"Illegal replay action: {path}")
                decisions += 1
            elif kind == "terminal":
                terminal = record
            else:
                raise ValueError(f"Unknown replay record: {path}")
    if not terminal or terminal["game"] != header["game"] or terminal["steps"] != decisions:
        raise ValueError(f"Incomplete replay: {path}")
    if (
        not decisions
        or terminal["result"] not in (0, 1, 2)
        or terminal["agent_errors"]
        or terminal["engine_errors"]
    ):
        raise ValueError(f"Failed arena game: {path}")
    return header, terminal


def decisions(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream decision records in their original replay order."""
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["type"] == "decision":
                yield record


def replay_schema(header: dict[str, Any]) -> str:
    """Return this agent's schema; mixed-deck replays carry one ID per seat."""
    names = {"mega_abomasnow", "mega_abomasnow_gbdt", "mega_abomasnow_transformer"}
    schemas = header.get("schemas", [header.get("schema_id")] * 2)
    if len(schemas) != 2:
        raise ValueError("Replay needs two seat schemas")
    own = [schemas[i] for i, name in enumerate(header["agents"]) if name in names]
    if not own or any(s != own[0] for s in own):
        raise ValueError("Missing/inconsistent agent schema")
    return own[0]
