# Directory Structure

## Package layout

- `btorch/models/`: stateful modules, neurons, synapses, RNNs, surrogate
  gradients, and parameter/state helpers.
- `btorch/models/neurons/`: neuron implementations; keep each cohesive model in
  one file and follow `lif.py`, `glif.py`, and `two_compartment.py`.
- `btorch/connectome/`: explicit and reversible connectome transforms.
- `btorch/analysis/`: numerical analysis and neuroscience metrics.
- `btorch/visualisation/`: plotting APIs; preserve the British spelling.
- `btorch/backend/` and `btorch/sparse/`: optimized and sparse implementations.
- `btorch/io/`, `btorch/datasets/`, and `btorch/utils/`: focused support code.

Tests mirror package areas under `tests/`. User-facing examples live in
`examples/`; English documentation lives in `docs/en/docs/`. Never edit the
generated `docs/en/docs/api` tree.

## Organization rules

- Put public research-library behavior in `btorch/`, not in an example or test.
- Add utilities to the closest existing domain package; avoid generic layers
  when a cohesive module is sufficient.
- Keep network models cohesive using the existing single-file/folder pattern.
- Update package `__init__.py` exports when adding public APIs.
- Use snake_case filenames and `test_<feature>.py` test modules.
- Keep experimental scripts and generated research output outside the package.

Representative modules include `btorch/models/neurons/lif.py`,
`btorch/connectome/connection.py`, and `btorch/sparse/matrices.py`.
