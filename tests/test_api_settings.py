"""Settings of the API: the key is required, checked, and never shown."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from invoice_recognition.api.settings import Settings, load_settings

ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No key in the environment and an empty working directory, whatever the developer's
    shell holds: each test writes the .env it needs."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def refusal(capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as stop:
        load_settings()
    assert not capsys.readouterr().out
    assert isinstance(stop.value.code, str)  # printed to stderr, exit status 1
    return stop.value.code


def test_a_missing_key_stops_the_start_and_names_the_variable(
    clean: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean / ".env").write_text("SOME_OTHER_SECRET=decoy-value-123\n")
    monkeypatch.setenv("ANOTHER_SECRET", "decoy-value-456")
    text = refusal(capsys)
    assert text == (
        "The API cannot start. OPENROUTER_API_KEY is not set. "
        "Settings are read from the environment and from a .env file. "
        "Copy .env.example to .env, fill it in and start the server again; "
        "with `fastapi dev`, stop it first with Ctrl+C."
    )


@pytest.mark.parametrize("key", ["", "   \n", "zzsecret words", "zzsecret-ключ", "zzsecret\tx"])
def test_an_unusable_key_stops_the_start_without_showing_it(
    key: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    text = refusal(capsys)
    assert text.startswith("The API cannot start. OPENROUTER_API_KEY is not valid: it must ")
    assert "zzsecret" not in text and "ключ" not in text


def test_a_dot_env_that_is_not_utf8_stops_the_start_cleanly(
    clean: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (clean / ".env").write_bytes(b"OPENROUTER_API_KEY=zzsecret\xff\xfe\n")
    text = refusal(capsys)
    assert text.startswith("The API cannot start. The .env file is not UTF-8 text. Copy ")
    assert "zzsecret" not in text


@pytest.mark.parametrize("key", ["", " ", "a b", "é"])
def test_an_unusable_key_is_refused_when_the_settings_are_built(key: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openrouter_api_key=SecretStr(key))


def test_the_key_is_read_from_the_environment_and_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-test-1\n")
    settings = load_settings()
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-test-1"
    assert "sk-or-test-1" not in repr(settings) and "sk-or-test-1" not in str(settings)


def test_the_key_is_read_from_dot_env_and_the_environment_wins(
    clean: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean / ".env").write_text("OPENROUTER_API_KEY=from-the-file\nUNRELATED=1\n")
    assert load_settings().openrouter_api_key.get_secret_value() == "from-the-file"
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-the-environment")
    assert load_settings().openrouter_api_key.get_secret_value() == "from-the-environment"


def test_no_dot_env_is_read_when_it_is_switched_off(clean: Path) -> None:
    (clean / ".env").write_text("OPENROUTER_API_KEY=from-the-file\n")
    with pytest.raises(ValidationError) as refused:
        Settings(_env_file=None)
    assert [e["type"] for e in refused.value.errors()] == ["missing"]


def test_every_setting_is_documented_in_env_example() -> None:
    lines = (ROOT / ".env.example").read_text().splitlines()
    documented = {s.split("=", 1)[0] for s in lines if s and not s.startswith("#")}
    assert documented == {name.upper() for name in Settings.model_fields}


def test_importing_the_package_needs_no_key() -> None:
    # main is the one module that reads the settings; nothing else imports it.
    code = "import invoice_recognition.api, invoice_recognition.api.app; print('ok')"
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert (done.returncode, done.stdout) == (0, "ok\n")


def test_the_served_app_refuses_to_start_without_a_key(clean: Path) -> None:
    # What `fastapi dev` / `fastapi run` do first: import the entry point.
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    done = subprocess.run(
        [sys.executable, "-c", "import invoice_recognition.api.main"],
        cwd=clean,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1
    assert done.stderr == (
        "The API cannot start. OPENROUTER_API_KEY is not set. "
        "Settings are read from the environment and from a .env file. "
        "Copy .env.example to .env, fill it in and start the server again; "
        "with `fastapi dev`, stop it first with Ctrl+C.\n"
    )
