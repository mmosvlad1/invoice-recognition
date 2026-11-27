"""The app the `fastapi` command serves (`[tool.fastapi]` in pyproject.toml).

Importing this module reads the settings from the environment and `.env`, and stops
the start when one is missing or invalid. Tests build their app with `create_app`.
"""

from invoice_recognition.api.app import create_app
from invoice_recognition.api.settings import load_settings

app = create_app(load_settings())
