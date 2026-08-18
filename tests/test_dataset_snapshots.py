"""Integrity checks for the exact benchmark snapshots used by the paper run."""
from hashlib import sha256
from pathlib import Path


def test_versioned_dataset_sha256_manifest_matches_files():
    data_dir = Path(__file__).parents[1] / "data"
    expected = {}
    for line in (data_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest

    assert expected == {"breast_cancer.csv": expected["breast_cancer.csv"], "wine.csv": expected["wine.csv"]}
    for filename, digest in expected.items():
        canonical_bytes = (data_dir / filename).read_bytes().replace(b"\r\n", b"\n")
        assert sha256(canonical_bytes).hexdigest() == digest
