"""Content identities and JSON files shared by command-line workflows."""

import hashlib
import json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
