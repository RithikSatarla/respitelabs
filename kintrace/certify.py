"""Signed GO / NO-GO records (ed25519).

A record is any JSON object (a recommission record from `kintrace incident`,
or a desk rig record). Signing adds a "signature" block:

  {"alg": "ed25519", "key_id": "...", "public_key": "<base64>", "sig": "<base64>"}

The signature covers the canonical JSON of the whole record except that
block (keys sorted, no spaces, UTF-8), including the "signed_at" time added
at signing. Changing any field, even deep inside, breaks it.

Anyone can make a key, so a valid signature only proves the record is
unchanged since it was signed by that key. `verify --pub` also checks the key
is the one you trust.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

SIG_KEY = "signature"


def canonical(record: dict) -> bytes:
    body = {k: v for k, v in record.items() if k != SIG_KEY}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _raw_pub(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(pub: Ed25519PublicKey) -> str:
    return hashlib.sha256(_raw_pub(pub)).hexdigest()[:16]


def init_keys(keydir: str, force: bool = False) -> tuple[str, str]:
    os.makedirs(keydir, exist_ok=True)
    priv_p, pub_p = os.path.join(keydir, "private.pem"), os.path.join(keydir, "public.pem")
    if os.path.exists(priv_p) and not force:
        raise FileExistsError(f"{priv_p} exists (use --force to replace it)")
    k = Ed25519PrivateKey.generate()
    fd = os.open(priv_p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()))
    with open(pub_p, "wb") as f:
        f.write(k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv_p, pub_p


def load_private(path: str) -> Ed25519PrivateKey:
    with open(path, "rb") as f:
        k = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(k, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an ed25519 key")
    return k


def load_public(path: str) -> Ed25519PublicKey:
    with open(path, "rb") as f:
        k = serialization.load_pem_public_key(f.read())
    if not isinstance(k, Ed25519PublicKey):
        raise ValueError(f"{path} is not an ed25519 key")
    return k


def sign(record: dict, key: Ed25519PrivateKey, signed_at: str | None = None) -> dict:
    r = {k: v for k, v in record.items() if k != SIG_KEY}
    r["signed_at"] = signed_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    pub = key.public_key()
    r[SIG_KEY] = dict(alg="ed25519", key_id=key_id(pub),
                      public_key=base64.b64encode(_raw_pub(pub)).decode(),
                      sig=base64.b64encode(key.sign(canonical(r))).decode())
    return r


def verify(record: dict, trusted: Ed25519PublicKey | None = None) -> tuple[bool, str]:
    """(ok, reason)."""
    s = record.get(SIG_KEY)
    if not isinstance(s, dict) or s.get("alg") != "ed25519":
        return False, "no ed25519 signature on this record"
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(s["public_key"]))
        sig = base64.b64decode(s["sig"])
    except Exception:
        return False, "signature block is malformed"
    if trusted is not None and _raw_pub(trusted) != _raw_pub(pub):
        return False, f"signed by key {key_id(pub)}, which is not the trusted key {key_id(trusted)}"
    if s.get("key_id") != key_id(pub):
        return False, "key id does not match the public key"
    try:
        pub.verify(sig, canonical(record))
    except (InvalidSignature, ValueError, TypeError):
        return False, "signature does not match: the record was changed after signing"
    return True, f"signed by key {key_id(pub)} at {record.get('signed_at', 'unknown time')}"
