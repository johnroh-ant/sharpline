"""Encrypted JSON files, so the ledger is private even though the repo is public."""
import gzip
import hashlib
import json
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import config

ROUNDS = 200_000
SALT = b"sharpline-ledger-v1"  # fixed so the dashboard derives the key once per visit
_keys = {}


def _key():
    p = os.environ.get("LEDGER_PASSPHRASE", "")
    if len(p) < 12:  # a missing Actions secret arrives as "", which would publish a readable ledger
        raise SystemExit("LEDGER_PASSPHRASE is missing or shorter than 12 characters")
    if p not in _keys:
        _keys[p] = hashlib.pbkdf2_hmac("sha256", p.encode(), SALT, ROUNDS, 32)
    return _keys[p]


def save(name, obj):
    path, nonce = config.DATA / f"{name}.enc", os.urandom(12)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(json.dumps(obj, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
    path.write_bytes(nonce + AESGCM(_key()).encrypt(nonce, raw, None))


def load(name, default=None):
    path = config.DATA / f"{name}.enc"
    if not path.exists():
        return default
    blob = path.read_bytes()
    return json.loads(gzip.decompress(AESGCM(_key()).decrypt(blob[:12], blob[12:], None)))
