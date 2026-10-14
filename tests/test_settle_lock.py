"""Two processes building one settled-body key: the second waits and reuses the body."""
import fcntl
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
import settled_body as sb


def _build(tmp_path):
    tools = SimpleNamespace(moddump=tmp_path / "moddump")
    return sb.build_settled_body(None, tmp_path, tmp_path / "t.setup", tmp_path / "ps", tmp_path / "p", tools, None)


def test_waiting_builder_reuses_body(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "settle_key", lambda spec, bins: "k")
    loaded = []
    monkeypatch.setattr(sb.SettledBody, "load", staticmethod(lambda p: loaded.append(Path(p)) or "BODY"))
    holder = open(tmp_path / "k.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX)  # another builder is mid-settle
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("body", _build(tmp_path)))
    th.start()
    th.join(0.5)
    assert th.is_alive()  # blocked on the lock, not settling a second copy
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "body.json").write_text("{}")
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    th.join(5)
    assert out["body"] == "BODY"
    assert loaded == [tmp_path / "k"]


def test_lock_falls_back_without_flock(tmp_path, monkeypatch, capsys):
    def no_flock(*_a):
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(sb.fcntl, "flock", no_flock)
    with sb._cache_lock(tmp_path, "k"):
        pass
    assert "without a lock" in capsys.readouterr().err
