"""Canonical, finite JSON fixture serialization."""
import json
from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
import numpy as np


def canonical_bytes(value):
    try:
        return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError('invalid JSON metadata') from exc


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_fixture(path, name, arrays, metadata):
    """Write name.json in path. metadata['units'] maps each array name to units."""
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('invalid fixture name')
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get('units'), Mapping) or not isinstance(arrays, Mapping):
        raise ValueError('metadata must contain units')
    units = metadata['units']
    if set(units) != set(arrays) or any(not isinstance(v, str) or not v for v in units.values()):
        raise ValueError('units must describe each array')
    encoded = {}
    for key, value in arrays.items():
        a = np.asarray(value)
        if not isinstance(key, str) or not key or a.dtype.kind not in 'biuf' or not np.all(np.isfinite(a)):
            raise ValueError('invalid fixture array')
        encoded[key] = {'dtype': a.dtype.name, 'shape': list(a.shape), 'data': a.tolist(), 'units': units[key]}
    data = canonical_bytes({'schema_version': 1, 'name': name, 'arrays': encoded, 'metadata': {k: v for k, v in metadata.items() if k != 'units'}})
    destination = Path(path) / (name + '.json')
    atomic_write(destination, data)
    return destination
