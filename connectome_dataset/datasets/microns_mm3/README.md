# microns_mm3

Metadata for the MICrONS mm3 connectome HDF5 payload.

## Identity

- Name: `microns_mm3`
- File: `microns_mm3_connectome.h5`
- Format: HDF5
- DVC pointer: `data/external/microns.dvc`
- Runtime path: `data/external/microns/microns_mm3_connectome.h5`

## Payload

The raw HDF5 file is large and must not be committed to git. It is DVC-managed
under `data/external/microns/`.

Restore it with:

```bash
dvc pull data/external/microns.dvc
```

## Loader

The historical loader requires `conntility`:

```python
import conntility

path = "data/external/microns/microns_mm3_connectome.h5"
full = conntility.ConnectivityMatrix.from_h5(path, "full")
condensed = conntility.ConnectivityMatrix.from_h5(path, "condensed")
```

`conntility` is not a core dependency of this package. Treat MICrONS loading as
an optional workflow unless a lightweight native loader is added.
