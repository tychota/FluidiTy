"""Independent immutable snapshots for caller-owned trial validation."""
from types import MappingProxyType
import numpy as np


class TrialRejected(Exception):
    """The caller's validator rejected a trial."""


class ReferenceState:
    def __init__(self, arrays, counters):
        copied = {}
        for name, value in arrays.items():
            a = np.asarray(value)
            if not isinstance(name, str) or not name or a.dtype.kind not in 'biuf' or not np.all(np.isfinite(a)):
                raise ValueError('invalid state array')
            a = np.array(a, copy=True, order='C')
            copied[name] = (a.tobytes(), a.dtype, a.shape)
        if any(not isinstance(k, str) or not k or isinstance(v, bool) or not isinstance(v, int) for k, v in counters.items()):
            raise ValueError('invalid counters')
        self._arrays = MappingProxyType(copied)
        self._counters = MappingProxyType(dict(counters))

    @property
    def arrays(self):
        return MappingProxyType({name: np.frombuffer(data, dtype=dtype).reshape(shape)
                                 for name, (data, dtype, shape) in self._arrays.items()})

    @property
    def counters(self):
        return self._counters

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError('state is immutable')
        object.__setattr__(self, name, value)


def trial_state(state, update):
    """Call update with mutable copies of the arrays and counters."""
    arrays = {k: v.copy() for k, v in state.arrays.items()}
    counters = dict(state.counters)
    update(arrays, counters)
    return ReferenceState(arrays, counters)


def accept_trial(committed, trial, validator):
    """Validate a trial without mutating either input; return a fresh snapshot."""
    if not validator(trial):
        raise TrialRejected('trial rejected')
    return ReferenceState(trial.arrays, trial.counters)
