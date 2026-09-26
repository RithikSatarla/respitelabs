"""Signed records: a signed record verifies, any change breaks it."""
import copy
import json

import pytest

from kintrace import certify

RECORD = {
    "status": "GO",
    "changed_at": "13:40:23",
    "findings": [{"fault": "camera_moved", "label": "Camera moved", "size": "21.2 mm, 0.11 deg"}],
    "fix": {"tcp_offset": [-0.0027, 0.024, 0.1838]},
    "test_picks": [True, True, True, True, True, True],
    "blockers": [],
}


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    d = tmp_path_factory.mktemp("keys")
    priv, pub = certify.init_keys(str(d))
    return certify.load_private(priv), certify.load_public(pub), str(d)


def _paths(obj, prefix=()):
    """Every leaf in a JSON object, as a path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _paths(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _paths(v, prefix + (i,))
    else:
        yield prefix


def _tamper(rec, path):
    r = copy.deepcopy(rec)
    o = r
    for p in path[:-1]:
        o = o[p]
    v = o[path[-1]]
    if isinstance(v, bool):
        o[path[-1]] = not v
    elif isinstance(v, (int, float)):
        o[path[-1]] = v + 1e-4
    else:
        o[path[-1]] = str(v) + "x"
    return r


def test_sign_and_verify(keys):
    priv, pub, _ = keys
    s = certify.sign(RECORD, priv)
    assert certify.verify(s)[0]
    assert certify.verify(s, pub)[0]
    # survives a round trip through a file
    assert certify.verify(json.loads(json.dumps(s, indent=2)), pub)[0]


def test_any_field_change_fails(keys):
    priv, pub, _ = keys
    s = certify.sign(RECORD, priv)
    paths = [p for p in _paths(s) if p[0] != "signature"]
    assert len(paths) > 10
    for p in paths:
        ok, why = certify.verify(_tamper(s, p), pub)
        assert not ok, f"changing {p} was not caught"


def test_added_or_removed_field_fails(keys):
    priv, _, _ = keys
    s = certify.sign(RECORD, priv)
    r = copy.deepcopy(s)
    r["note"] = "added later"
    assert not certify.verify(r)[0]
    r = copy.deepcopy(s)
    del r["blockers"]
    assert not certify.verify(r)[0]


def test_signature_block_tamper_fails(keys):
    priv, pub, _ = keys
    s = certify.sign(RECORD, priv)
    for k in ("sig", "public_key", "key_id"):
        r = copy.deepcopy(s)
        r["signature"][k] = r["signature"][k][:-4] + "AAAA"
        assert not certify.verify(r, pub)[0], k
    r = copy.deepcopy(s)
    del r["signature"]
    assert not certify.verify(r)[0]


def test_other_key_rejected_with_trusted_key(keys, tmp_path):
    _, pub, _ = keys
    priv2, _ = certify.init_keys(str(tmp_path / "other"))
    s = certify.sign(RECORD, certify.load_private(priv2))
    assert certify.verify(s)[0]  # valid by its own key
    ok, why = certify.verify(s, pub)  # but not the key we trust
    assert not ok and "not the trusted key" in why


def test_cli_roundtrip(keys, tmp_path, capsys):
    from kintrace.cli import main

    _, _, d = keys
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(RECORD))
    main(["certify", str(p), "--key", f"{d}/private.pem"])
    main(["verify", str(p), "--pub", f"{d}/public.pem"])
    assert "VALID" in capsys.readouterr().out
    r = json.loads(p.read_text())
    r["status"] = "NO-GO" if r["status"] == "GO" else "GO"
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit):
        main(["verify", str(p), "--pub", f"{d}/public.pem"])
    assert "INVALID" in capsys.readouterr().out
