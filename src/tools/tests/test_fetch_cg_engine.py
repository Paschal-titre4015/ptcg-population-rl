"""Check byte preservation and refusal to overwrite altered upstream assets."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_cg_engine import install, safe_relative, unpack_one, verify_tree


class EngineDistributionTests(unittest.TestCase):
    """Validate engine distribution tests."""

    def setUp(self) -> None:
        """Create an isolated fixture for each test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = b"\xef\xbb\xbf// official bytes\r\n#pragma once\r\n"
        self.manifest = {
            "files": {
                "All.h": {
                    "size": len(self.payload),
                    "sha256": hashlib.sha256(self.payload).hexdigest(),
                }
            }
        }

    def test_plain_download_preserves_bom_and_crlf(self) -> None:
        """Check plain download preserves bom and crlf."""
        source = self.root / "download"
        source.mkdir()
        (source / "All.h").write_bytes(self.payload)
        target = self.root / "All.h"
        unpack_one(source, target, Path("All.h"), "ptcg_engine/All.h")
        self.assertEqual(target.read_bytes(), self.payload)

    def test_zip_download_preserves_bom_and_crlf(self) -> None:
        """Check zip download preserves bom and crlf."""
        source = self.root / "download"
        source.mkdir()
        with zipfile.ZipFile(source / "All.h.zip", "w") as archive:
            archive.writestr("ptcg_engine/All.h", self.payload)
        target = self.root / "All.h"
        unpack_one(source, target, Path("All.h"), "ptcg_engine/All.h")
        self.assertEqual(target.read_bytes(), self.payload)

    def test_unsafe_zip_entry_is_not_extracted(self) -> None:
        """Check unsafe zip entry is not extracted."""
        source = self.root / "download"
        source.mkdir()
        with zipfile.ZipFile(source / "All.h.zip", "w") as archive:
            archive.writestr("../All.h", self.payload)
        with self.assertRaises(ValueError):
            unpack_one(source, self.root / "target", Path("All.h"), "ptcg_engine/All.h")
        self.assertFalse((self.root / "target").exists())
        for name in ("../All.h", "/All.h", "x/../../All.h", "x\\All.h"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)

    def test_modified_installation_is_not_overwritten(self) -> None:
        """Check modified installation is not overwritten."""
        destination = self.root / "cg"
        destination.mkdir()
        changed = self.payload + b"// local edit\n"
        (destination / "All.h").write_bytes(changed)
        with self.assertRaises(ValueError):
            install(self.manifest, destination, self.root / "cache")
        self.assertEqual((destination / "All.h").read_bytes(), changed)

    def test_extra_files_and_symlinks_are_rejected(self) -> None:
        """Check extra files and symlinks are rejected."""
        destination = self.root / "cg"
        destination.mkdir()
        (destination / "All.h").write_bytes(self.payload)
        verify_tree(destination, self.manifest)
        extra = destination / "patch.h"
        extra.write_bytes(b"extra")
        with self.assertRaises(ValueError):
            verify_tree(destination, self.manifest)
        extra.unlink()
        extra.symlink_to(destination / "All.h")
        with self.assertRaises(ValueError):
            verify_tree(destination, self.manifest)


if __name__ == "__main__":
    unittest.main()
