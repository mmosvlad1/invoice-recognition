"""The HTTP API: upload an invoice, follow its reading, fetch the result.

`create_app(settings)` builds the app; `invoice_recognition.api.main` is the module
the `fastapi` command serves, the only one that reads the settings from the
environment. Importing this package reads nothing and needs no key.
"""
