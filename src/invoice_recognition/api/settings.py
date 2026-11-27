"""Settings of the API: read from the environment, then from a `.env` file in the
working directory. A variable set in the environment wins over the file.

Every setting is documented in `.env.example`.
"""

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Other variables in the environment or in .env are not ours: ignored, never an error.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: SecretStr = Field(
        description="The OpenRouter API key every model request is sent with. Required."
    )

    @field_validator("openrouter_api_key")
    @classmethod
    def _usable_as_a_header(cls, value: SecretStr) -> SecretStr:
        """The rules of `llm.openrouter_client`, checked at start-up: surrounding whitespace
        is stripped (a key pasted into a file often ends with a newline); what is left must
        not be empty and may hold only printable ASCII without spaces. The message never
        shows the key."""
        key = value.get_secret_value().strip()
        if not key or not all("!" <= c <= "~" for c in key):
            raise PydanticCustomError(
                "api_key_format",
                "it must not be empty and may contain only printable ASCII characters, "
                "without spaces",
            )
        return SecretStr(key)


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
            else:
                problems.append(f"{name} is not valid: {error['msg']}.")
        problems.append("Settings are read from the environment and from a .env file.")
        text = " ".join(["The API cannot start.", *problems, RESTART])
        # `from None`: the chained ValidationError would print the value it was given.
        raise SystemExit(text) from None
