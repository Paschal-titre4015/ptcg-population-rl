"""Fetch the official C++ engine, verify its bytes, and install without patches."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION = {
    "competition": "pokemon-tcg-ai-battle",
    "source_prefix": "ptcg_engine/ptcgProgram 22/",
    "files": {
        "ActivateInfo.h": {
            "size": 1714,
            "sha256": "a4c327b6c8fc6be3a80e67eae90f07e13ab799af65b3f6bb4456b387aeacedf8",
        },
        "AddLog.h": {
            "size": 6906,
            "sha256": "70c4b980fd7b64fd53e4add3269de076546b8f3284304d6cca84b6b070aef869",
        },
        "AddOption.h": {
            "size": 4268,
            "sha256": "dfe24ced984da07aaf50cab7835744064c19de87e24c14a943fb01584c15117f",
        },
        "All.h": {
            "size": 748,
            "sha256": "b176e6c5d1d3dde03e103f67b26117cab8e89b607bcfea26e01147be6e9dd41f",
        },
        "Api.h": {
            "size": 8427,
            "sha256": "786acae884631bdcbab0311471316bc75d62acb465b8011bf09dad05628bcafd",
        },
        "ApiData.h": {
            "size": 933,
            "sha256": "b21025cb1027556746f11fd73d6851ef231fa6492d82dd8074e80d70cbf93da9",
        },
        "ApiJson.h": {
            "size": 13028,
            "sha256": "fa405a0421ad09fbae6e3922bd73a0a07333b17790dea7e8fd90c1b97f326851",
        },
        "ApiType.h": {
            "size": 3130,
            "sha256": "d270162140ce1e518a9f2ea6a4a73ed10e9585b8a212eb9a3eca0ba1c220d655",
        },
        "BattleData.h": {
            "size": 8844,
            "sha256": "6aa12c3cdec1b3e3aca0b4c102952ab5713a4ba10c1a0a8df66be3b42db5db8f",
        },
        "Binary.h": {
            "size": 4934,
            "sha256": "ab42f329818f19e4a03d76edf8066312fc0060bbc1b5524a17897a1636b12257",
        },
        "Card.h": {
            "size": 11962,
            "sha256": "32a3b5fc5b1cb2eb7826b97faade4ec36ed5aa83082bab32e6ad8fbb394c844d",
        },
        "CardImpl.h": {
            "size": 878383,
            "sha256": "286a51820d36f9b60b5b13c58bc2edff352eb4050581ddadf1960f13fd6f21a9",
        },
        "CardMove.h": {
            "size": 16529,
            "sha256": "7252d27c63c539c30899e3c7136e34506e9b3be3f09e974cfc4b2f3139e26a0d",
        },
        "Core.h": {
            "size": 1524,
            "sha256": "fdc29455564073b865cc8ac0e0429e66a1df4a245594b26f939d64bc1d0fd7a3",
        },
        "CreateCard.h": {
            "size": 57440,
            "sha256": "54bf6a7fc374f740454116b5a291460192287b8d1ba8791cf223ecda925f78d0",
        },
        "EffectContinual.h": {
            "size": 16472,
            "sha256": "5ac571b4ed6e5b324098cd9ca0377f53eba4788172ab8d9c891648d4c5b09c83",
        },
        "EffectInstant.h": {
            "size": 65977,
            "sha256": "e1729b7f5ad4f928fd7b0cb6053be3b935065711483f7018797ba822679e46f8",
        },
        "EffectProc.h": {
            "size": 36883,
            "sha256": "ab5aa982826910d372b5983308f995675ff2c9b40823b444fbff6f1a26f0f975",
        },
        "Export.cpp": {
            "size": 5117,
            "sha256": "1269f64671527d50b8560540473438e9781926fdbdb0c9383b344f0fb91bf82f",
        },
        "FixedList.h": {
            "size": 4907,
            "sha256": "2e1de1418edf279040ba7dc9e8ec9fb1608880f6a66d742b5036b74f1f31836f",
        },
        "Framework.h": {
            "size": 903,
            "sha256": "f0ad603d615ba1769c622982e791d042a900d4dfa0b2d53f6ce48b84eced19a2",
        },
        "Game.h": {
            "size": 2857,
            "sha256": "1dd43b6a46f225ff2a238d92e8d718cf0d7414773bfe52388f5325bf02b0a097",
        },
        "GameFunction.h": {
            "size": 1536,
            "sha256": "b88525cdd5a760d7c8871d99897453a2433fdd024f76a67ea8568806f4cfcf57",
        },
        "GameProc.h": {
            "size": 30790,
            "sha256": "cf1537be5c439bed44fbf08b91a6b066f3878ee5ae8a88a828f110bce01b3260",
        },
        "GameUtil.h": {
            "size": 6201,
            "sha256": "ae42efb5ec81b9292eeeb51742de9661af8a6cc0a7e0e0999ff0dea347d55aab",
        },
        "InitializeCard.h": {
            "size": 1211,
            "sha256": "130b20dbedeb8de09e4cab1b3cba0b12fe1d0654788b134e29d2245dd0f36da8",
        },
        "JsonBuilder.h": {
            "size": 3500,
            "sha256": "4a6c1604e4fce3669a7734c0c80a41227448af9ad7f9c5834757d7d1fb143fe7",
        },
        "LICENSES/LicenseRef-PTCG-ABC-Competition-Use-Only.txt": {
            "size": 1590,
            "sha256": "0bc3060e440432fa9481071ee9531d0d34e7212b1b6ffe595e38b9c9cd9c081b",
        },
        "PlayerState.h": {
            "size": 3713,
            "sha256": "269e7c4a2c1522245fc2ddc1e89cf83412d1b00f1deed4b673f3275906a412a1",
        },
        "PullTrigger.h": {
            "size": 5310,
            "sha256": "620076e485690bdab1f82327bc862374803794b54c5cccce0339a5abc0f8db99",
        },
        "README.md": {
            "size": 2293,
            "sha256": "7e49513e5704a95f13772d87e0e955eba20903cd7184367928e3a5f08540c91d",
        },
        "REUSE.toml": {
            "size": 464,
            "sha256": "9bcfaf05c5d0a6abfe923bcd918c8fd9524c297e44592085db70a67639f9ae5b",
        },
        "SatisfyCondition.h": {
            "size": 9988,
            "sha256": "5a26309ff23e616b6a27c66e5a9125d6062763cd5dbf0e903f589a3f85a59127",
        },
        "Search.h": {
            "size": 6321,
            "sha256": "d73553ee85dd993f2c1d7e4606e3e38f988f5390b88080148f4ef98a926762f9",
        },
        "SelectProc.h": {
            "size": 14926,
            "sha256": "cefae973a571d00a7c34a94f613e27c5048080c371ee62824dca390519099c46",
        },
        "SetProperty.h": {
            "size": 14367,
            "sha256": "a563762c4a8f6c1f9c77048b89f770cd2e7a5d4801cd29256196dda9e40fcb81",
        },
        "SetupProc.h": {
            "size": 8266,
            "sha256": "fa1e3290d45796fa802b50f3005c657202653173183ca0abb024656b0ed7e1a9",
        },
        "Skill.h": {
            "size": 8400,
            "sha256": "53415b8143e0b23dbfe6b79997924effd90afa6104022334c6e8aeaa3c23991e",
        },
        "State.h": {
            "size": 49492,
            "sha256": "a12bc5669b4b79c122899142f86616f8d9926684d9029392220c5c886de2a33c",
        },
        "TargetList.h": {
            "size": 26064,
            "sha256": "c93bea62ea3de7b733e3cbbc8d75cde9a612c3dd956516597a8fe8858dc8eac7",
        },
        "ToJson.h": {
            "size": 9993,
            "sha256": "84ee63939863493520ebe29e8ca717217ebf90191829ea80d1349942aa867602",
        },
        "Types.h": {
            "size": 36076,
            "sha256": "c5a9ad65b1221fc9b6ed09e5f56c6bd7ec325117cc2247592164abe698bad0a8",
        },
        "game.sln": {
            "size": 945,
            "sha256": "05a0d8ad94f1aef75733d18dc354534aa974437e405825eef5bfa9cc2dfec3cc",
        },
        "game.vcxproj": {
            "size": 6161,
            "sha256": "afb079173a8305e0441b95cef7a304ddd799a88fabb656a05ed541764841e6eb",
        },
        "game.vcxproj.filters": {
            "size": 5100,
            "sha256": "f32e46fa571c00c29051f90ba08f16f06812a24d40c889dd4ac0a6747eac22d7",
        },
    },
}


def safe_relative(name: str) -> PurePosixPath:
    """Reject absolute paths, parent traversal, and ambiguous archive separators."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe distribution path: {name}")
    return Path(*path.parts)


def verify_file(path: Path, spec: dict[str, Any]) -> None:
    """Verify a distribution file's size and SHA-256 digest."""
    data = path.read_bytes()
    if len(data) != spec["size"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"Official file changed or corrupted: {path}")


def verify_tree(directory: Path, manifest: dict[str, Any]) -> None:
    """Require exactly the original distribution files and reject symlinks."""
    expected = set(manifest["files"])
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in engine: {path}")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected:
        raise ValueError(
            f"Engine file list mismatch: missing={expected - actual}, extra={actual - expected}"
        )
    for name, spec in manifest["files"].items():
        verify_file(directory / safe_relative(name), spec)


def unpack_one(download_dir: Path, target: Path, relative: PurePosixPath, remote_name: str) -> None:
    """Kaggle may serve an individual file either directly or as a ZIP."""
    direct = download_dir / relative.name
    if direct.is_file():
        shutil.copyfile(direct, target)
        return
    archives = list(download_dir.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Unexpected Kaggle download layout: {download_dir}")
    with zipfile.ZipFile(archives[0]) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1:
            raise ValueError("Expected exactly one file in Kaggle download ZIP")
        member = members[0]
        safe_relative(member.filename)
        if member.filename not in (relative.name, relative.as_posix(), remote_name):
            raise ValueError(f"Unexpected ZIP entry: {member.filename}")
        # Read bytes only; never apply archive paths, permissions, or symlinks.
        target.write_bytes(archive.read(member))


def install(
    manifest: dict[str, Any], destination: Path, cache: Path, verify_only: bool = False
) -> None:
    """Download into staging and install the verified engine atomically."""
    for name in manifest["files"]:
        safe_relative(name)
    if destination.is_symlink():
        raise ValueError(f"Engine destination must not be a symlink: {destination}")
    if destination.exists():
        verify_tree(destination, manifest)
        print(f"Verified {len(manifest['files'])} unchanged engine files: {destination}")
        return
    if verify_only:
        raise FileNotFoundError(f"Engine has not been downloaded: {destination}")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cg-staging-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for index, (name, spec) in enumerate(manifest["files"].items(), 1):
            relative = safe_relative(name)
            cached = cache / "files" / relative
            if cached.exists():
                verify_file(cached, spec)
            else:
                cache.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache) as download:
                    remote = manifest["source_prefix"] + relative.as_posix()
                    api.competition_download_file(
                        manifest["competition"], remote, path=download, quiet=True
                    )
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    candidate = Path(download) / "verified-payload"
                    unpack_one(Path(download), candidate, relative, remote)
                    verify_file(candidate, spec)
                    candidate.replace(cached)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
            print(f"{index}/{len(manifest['files'])} {name}", flush=True)
        verify_tree(staging, manifest)
        if destination.exists():
            raise FileExistsError(f"Destination appeared during download: {destination}")
        staging.rename(destination)
    print(f"Installed unmodified competition engine: {destination}")


def main() -> None:
    """Parse command-line arguments and run this tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "src/engine/cg")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "external/engine_download")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    install(DISTRIBUTION, args.destination, args.cache_dir, args.verify_only)


if __name__ == "__main__":
    main()
