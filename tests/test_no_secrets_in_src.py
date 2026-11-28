"""No file that looks like a secret lives under src/: everything there is packaged, and the
files under web/static are served to anyone. The image's .dockerignore keeps such files
out too; this test stops one from being committed (or left there) in the first place."""

from pathlib import Path

SRC = Path(__file__).parent.parent / "src"


def _looks_secret(path: Path) -> bool:
    name = path.name.lower()
    return (
        (name.startswith(".env") and name != ".env.example")
        or name.startswith("id_rsa")
        or name.endswith((".pem", ".key"))
    )


def test_no_secret_looking_file_under_src() -> None:
    assert SRC.is_dir()
    found = sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*") if _looks_secret(p))
    assert found == []


def test_the_rule_catches_what_it_should() -> None:
    caught = [".env", ".env.local", "id_rsa", "id_rsa.pub", "server.pem", "tls.KEY"]
    passed = [".env.example", "environment.py", "keys.py", "invoice.pdf"]
    assert all(_looks_secret(Path(n)) for n in caught)
    assert not any(_looks_secret(Path(n)) for n in passed)
