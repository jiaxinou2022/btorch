# Quality Guidelines

## Required patterns

- Target Python 3.10+ and use modern annotations (`X | Y`, `list`, `dict`).
- Use `jaxtyping` when tensor shape annotations materially clarify an API.
- Preserve device, dtype, and batch/time semantics across operations.
- Follow existing stateful `torch.nn.Module` patterns: parameters remain
  parameters; persistent state uses buffers or `register_memory`.
- Preserve `torch.compile` compatibility and ONNX friendliness where relevant.
- Use Google-style docstrings. Put constructor parameters in the class docstring
  and mathematical models in `.. math::` blocks.
- Keep code, comments, and docstrings within 88 characters.

## Avoid

- Hidden tensor copies or device transfers.
- Python-side mutation that breaks compiled execution without documentation.
- Heavy new dependencies for functionality covered by existing dependencies.
- Broad refactors mixed into a focused feature or bug fix.
- Editing generated documentation under `docs/en/docs/api`.

## Verification

- Add tests under the matching `tests/` subtree and explain non-obvious setup.
- Run targeted pytest first, then `pytest tests` when practical.
- Run `ruff check .`; use pre-commit before a pull request.
- For compile-sensitive changes, include relevant compile or ONNX tests.
- Update `docs/en/docs/` for user-facing behavior and `README.md` for
  installation or workflow changes.

Review shape/dtype/device correctness, state registration, scientific
semantics, documentation, regression coverage, and reversible transforms.
