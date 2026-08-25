# Logging Guidelines

Btorch is a library, so it must not configure global logging or print routine
diagnostics to stdout.

- Prefer explicit return values and exceptions over logging.
- Use Python logging warnings only for supported fallbacks or behavior users
  should inspect; `btorch/models/functional.py` has existing examples.
- Never log full tensors, connectomes, credentials, or large datasets.
- Keep messages actionable and identify the selected fallback or unsupported
  condition.
- CLI and example scripts may report progress, but reusable package modules
  should remain quiet during normal operation.

Do not add a logging dependency solely for package diagnostics.
