# Data Storage Guidelines

Btorch is a scientific Python library and has no database, ORM, or migration
layer. Do not introduce one unless a feature explicitly requires it.

- Keep tensors on the caller's device and preserve dtype unless conversion is
  part of the documented API.
- Make batch, time, neuron, and feature dimensions explicit in validation,
  annotations, and docstrings.
- Use `register_buffer` or the established `register_memory` mechanism for
  module state. See `btorch/models/neurons/lif.py` and
  `btorch/models/synapse.py`.
- Keep connectome conversions explicit and reversible; validate DataFrame and
  sparse-array inputs before transforming them.
- Use existing helpers under `btorch/io/` and `btorch/utils/` for supported
  formats.
- Avoid implicit CPU copies, dtype widening, or NumPy round trips in Torch code.

When persistent formats change, preserve backward readability where practical
and add round-trip tests under `tests/io/` or the owning package's test folder.
