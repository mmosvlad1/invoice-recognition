"""The prompts are part of the measured behaviour: a change to one is a change of results."""

import hashlib

import pytest

from invoice_recognition.recognition import prompts

# sha256 of each prompt as it was measured. A deliberate change of a prompt updates the
# value here, in the same commit, after the change has been measured.
MEASURED = {
    "LINE_RULES": "c46b8db64f934904c763f1f8a0f3ca6319e9131dcfc97ed2dd28dd84f6e5e4f0",
    "EXTRACT_PAGE": "9da1785e7a5351b019ab810ad89191d84a683083cf0538ed5d5108e932324916",
    "EXTRACT_HEADER": "b2c61944f4d3fdcff2b76bc46581cb38c94f48359876ec0af06957e0875c3fa2",
}


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_prompt_is_the_measured_one(name: str) -> None:
    text: str = getattr(prompts, name)
    assert hashlib.sha256(text.encode()).hexdigest() == MEASURED[name]


def test_page_prompt_takes_the_page_and_nothing_else() -> None:
    text = prompts.EXTRACT_PAGE.format(page=2, total=5)
    assert text.startswith("You read ONE page of a supplier invoice: page 2 of 5.")
    assert "{page}" not in text and "{{" not in text and '{"rows": []}' in text
    # The page prompt keeps the line rules, with control totals reported instead of left out.
    assert "must NOT be returned as line items (report them as control totals instead):" in text
    assert "must be left out:" in prompts.LINE_RULES and "must be left out:" not in text
