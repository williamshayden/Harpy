"""The shared file layer has no task or training dependencies."""

import hashlib
import subprocess
import sys

import pytest

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)


def test_neutral_import_does_not_load_task_or_learning_modules():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import harpy._serialization; "
            "assert not any(name.startswith(('harpy.learning', 'harpy.experiments', "
            "'harpy.tasks', 'numpy', 'gymnasium', 'torch', 'stable_baselines3')) "
            "for name in sys.modules)",
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr.decode()
    assert process.stdout == process.stderr == b""


def test_canonical_bytes_preserve_historical_encoding():
    assert canonical_json_bytes({"z": [True, None, 2.5], "a": "caf\u00e9"}) == (
        b'{"a":"caf\\u00e9","z":[true,null,2.5]}\n'
    )


@pytest.mark.parametrize(
    "content, message",
    [
        (b'{"x":{"a":1,"a":2}}', "duplicate"),
        (b'{"x":[NaN]}', "finite"),
        (b'{"x":[Infinity]}', "finite"),
        (b'{"x":[-Infinity]}', "finite"),
        (b'{"x":[1e999]}', "finite"),
        (b'{"x":{"y":-1e999}}', "finite"),
        (b'\xff{"x":1}', "UTF-8"),
        (b"[]", "object"),
        (b"null", "object"),
        (b"{", "JSON"),
    ],
)
def test_strict_decoder_rejects_ambiguous_documents(content, message):
    with pytest.raises(ValueError, match=message):
        decode_json_bytes(content)


def test_incremental_hash_matches_complete_payload(tmp_path):
    payload = b"multi-block payload" * 130_000
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_create_only_writer_rejects_existing_symlink(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"preserved")
    destination = tmp_path / "result.json"
    destination.symlink_to(original)
    with pytest.raises(FileExistsError):
        write_new_bytes(destination, b"new")
    assert destination.is_symlink()
    assert original.read_bytes() == b"preserved"
    assert set(tmp_path.iterdir()) == {original, destination}
