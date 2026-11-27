"""The page's script modules, tested with Node's own test runner: no browser, no packages
(tests/js/*.test.mjs). Node 20 or newer is needed. Without it the tests are skipped with
a visible reason on a developer's machine, and fail where CI is set: a check that
silently does not run is worse than none.

Before Node runs, real API answers are written to a directory the script tests read
(API_SAMPLES_DIR): the hostile test puts a payload into every value of them and checks
that no builder lets it through as markup."""

import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import ImageFilter

import demo_server
import drawn
from fakes import Handler, Server, header_for, line, never, serve, wait_final

NODE_MIN = 20
MISSING = f"Node {NODE_MIN}+ not found: the page's script tests did not run"
JS_TESTS = sorted((Path(__file__).parent / "js").glob("*.test.mjs"))


def node() -> str | None:
    """The path of a Node new enough, or None."""
    path = shutil.which("node")
    if path is None:
        return None
    version = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
    found = re.match(r"v(\d+)\.", version.stdout)
    return path if found and int(found.group(1)) >= NODE_MIN else None


async def _read(directory: Path, name: str, data: bytes, server: Handler) -> None:
    async with serve(server) as (client, _):
        upload = await client.post("/invoices", files={"file": (name, data)})
        job = await wait_final(client, upload.json()["id"])
        items = (await client.get("/invoices")).json()["items"]
    (directory / f"{name}.job.json").write_text(json.dumps(job))
    (directory / f"{name}.item.json").write_text(json.dumps(items[0]))


async def _write_answers(directory: Path) -> None:
    small = demo_server.SAMPLES / "invoice-small-misprint.pdf"
    rows = [line(n) for n in range(1, 4)]
    blurred = drawn.photo().filter(ImageFilter.GaussianBlur(3))
    readers = demo_server.readers()
    await _read(
        directory, "small.pdf", small.read_bytes(), readers[demo_server.digest(small.read_bytes())]
    )
    await _read(
        directory, "scan.pdf", drawn.scanned("good", "blank"), Server(header_for(rows), {1: [rows]})
    )
    await _read(directory, "photo.jpg", drawn.jpeg(blurred), never)


def test_the_page_scripts(tmp_path: Path) -> None:
    path = node()
    if path is None:
        if os.environ.get("CI"):
            pytest.fail(MISSING)
        pytest.skip(MISSING)
    assert JS_TESTS
    asyncio.run(_write_answers(tmp_path))
    env = os.environ | {"API_SAMPLES_DIR": str(tmp_path)}
    run = subprocess.run(
        [path, "--test", *map(str, JS_TESTS)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    if run.returncode != 0:
        pytest.fail(f"node --test failed:\n{run.stdout}\n{run.stderr}", pytrace=False)
