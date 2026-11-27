"""Settings of the API: read from the environment, then from a `.env` file in the
working directory. A variable set in the environment wins over the file.

Every setting is documented in `.env.example`.
"""

from typing import Self

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict

from invoice_recognition.recognition.config import FAST, MAX_IMAGE_PIXELS

MIN_ACCESS_CODE = 16
"""The shortest access code accepted. `secrets.token_urlsafe(24)` gives 32 characters."""


def _printable(text: str) -> bool:
    return all("!" <= c <= "~" for c in text)


class Settings(BaseSettings):
    # Other variables in the environment or in .env are not ours: ignored, never an error.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: SecretStr = Field(
        description="The OpenRouter API key every model request is sent with. Required."
    )
    access_code: SecretStr = Field(
        default=SecretStr(""),
        description="The code every /invoices request must carry (Authorization: Bearer). "
        "Empty: no check, for local development only.",
    )

    # The checks of an upload.
    max_file_bytes: int = Field(
        default=10 * 1024 * 1024, ge=1, description="The largest file an upload may carry."
    )
    max_pages: int = Field(default=50, ge=1, description="The most pages a PDF may have.")
    max_image_pixels: int = Field(
        default=MAX_IMAGE_PIXELS,
        ge=1,
        description="The most pixels an image may have: an uploaded photo, or the images of "
        "one page of a scan.",
    )

    # The reading of documents.
    max_documents: int = Field(
        default=10,
        ge=1,
        description="Documents waiting or being read at once; one more is refused.",
    )
    reading_at_once: int = Field(
        default=3, ge=1, description="Of those, how many are read at the same time."
    )
    model_requests_in_flight: int = Field(
        default=24, ge=1, description="Model requests in flight at once, for all documents."
    )
    deadline_s: float = Field(
        default=240.0, gt=0, description="Seconds one document may take, from the start of reading."
    )
    jobs_kept: int = Field(
        default=50,
        ge=1,
        description="Documents the process keeps; older finished ones are dropped.",
    )
    page_model: str = Field(default=FAST, min_length=1, description="Reads every page.")
    header_model: str = Field(default=FAST, min_length=1, description="Reads the header.")
    layout_model: str = Field(
        default=FAST, min_length=1, description="Describes the table of a PDF with a text layer."
    )

    @field_validator("openrouter_api_key")
    @classmethod
    def _usable_as_a_header(cls, value: SecretStr) -> SecretStr:
        """The rules of `llm.openrouter_client`, checked at start-up: surrounding whitespace
        is stripped (a key pasted into a file often ends with a newline); what is left must
        not be empty and may hold only printable ASCII without spaces. The message never
        shows the key."""
        key = value.get_secret_value().strip()
        if not key or not _printable(key):
            raise PydanticCustomError(
                "api_key_format",
                "it must not be empty and may contain only printable ASCII characters, "
                "without spaces",
            )
        return SecretStr(key)

    @field_validator("access_code")
    @classmethod
    def _usable_in_a_header(cls, value: SecretStr) -> SecretStr:
        """Stripped like the key. Empty switches the check off; otherwise at least
        MIN_ACCESS_CODE characters (the address of the service is public, so a short code
        could be guessed) of printable ASCII without spaces, so that any client (a
        browser's fetch included) can send it in a header exactly as it is written. The
        message never shows the code."""
        code = value.get_secret_value().strip()
        if code and (len(code) < MIN_ACCESS_CODE or not _printable(code)):
            raise PydanticCustomError(
                "access_code_format",
                "it must be at least {length} characters long and may contain only printable "
                "ASCII characters, without spaces; leave it empty to switch the check off",
                {"length": MIN_ACCESS_CODE},
            )
        return SecretStr(code)

    @model_validator(mode="after")
    def _room_for_every_unfinished_job(self) -> Self:
        """A job is never dropped before it is finished, so the store must be able to hold
        every document that may be waiting or being read."""
        if self.jobs_kept < self.max_documents:
            raise PydanticCustomError(
                "jobs_kept_too_small",
                "JOBS_KEPT ({kept}) must not be smaller than MAX_DOCUMENTS ({documents})",
                {"kept": self.jobs_kept, "documents": self.max_documents},
            )
        return self


# The last sentence of every refusal to start: what to do about it.
RESTART = (
    "Copy .env.example to .env, fill it in and start the server again; "
    "with `fastapi dev`, stop it first with Ctrl+C."
)


def load_settings() -> Settings:
    """The settings of the running app. A missing or invalid setting stops the start with
    a message that names the variables and never shows their values."""
    try:
        return Settings()
    except UnicodeDecodeError:  # .env holds bytes that are not UTF-8: say so, not how
        raise SystemExit(
            f"The API cannot start. The .env file is not UTF-8 text. {RESTART}"
        ) from None
    except ValidationError as e:
        problems = []
        for error in e.errors(include_input=False, include_url=False, include_context=False):
            name = ".".join(str(part) for part in error["loc"]).upper()
            if error["type"] == "missing":
                problems.append(f"{name} is not set.")
            elif not name:  # a rule between settings: its message names them
                problems.append(f"{error['msg']}.")
            else:
                problems.append(f"{name} is not valid: {error['msg']}.")
        problems.append("Settings are read from the environment and from a .env file.")
        text = " ".join(["The API cannot start.", *problems, RESTART])
        # `from None`: the chained ValidationError would print the value it was given.
        raise SystemExit(text) from None
