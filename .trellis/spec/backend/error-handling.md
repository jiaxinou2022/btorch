# Error Handling

- Validate public inputs near the API boundary, especially tensor rank, shape,
  dtype, device compatibility, index ranges, and mutually exclusive options.
- Raise `ValueError` for invalid values or incompatible shapes and include the
  received value or shape when useful. See `btorch/sparse/events.py` and
  `btorch/backend/persistent_snn.py`.
- Use `TypeError` when the Python object type itself is unsupported.
- Do not silently move tensors, coerce unsupported dtypes, or repair malformed
  structural data.
- Preserve the original exception with `raise ... from exc` when translating a
  low-level parsing or conversion failure.
- Warn for a supported fallback; conditions that could invalidate a scientific
  result must fail.

Tests should assert the exception type and invalid condition. See
`tests/connectome/test_delay_expansion.py` for validation examples.
