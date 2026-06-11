"""Experimental-data registry and back-calculation uncertainties for the X-EISD scorer."""
import os
import numpy as np
from glob import glob


class Stack:
    def __init__(self, name, data, sigma=None, mu=None):
        self.name = name
        self.data = data
        self.sigma = sigma
        self.mu = mu

    def add(self, new_stack):
        self.data = np.concatenate([self.data, new_stack.data], axis=0)


# back-calculation uncertainties (cs per-atom from UCBShifts; jc set in read_bc_data)
BC_ERRORS = {
    'pre': 0.001, 'noe': 0.01, 'fret': 0.0074, 'rh': 0.812, 'rdc': 0.88,
    'cs': {'C': 1.31, 'CA': 0.97, 'CB': 1.26, 'H': 0.38, 'HA': 0.29, 'N': 2.16},
}

# Exp data dir: env IDPFORGE_EXP_DATA, else ../../Data/exp. Resolved lazily (see _LazyExpDataLib).
EXP_PATH_BASE = os.environ.get("IDPFORGE_EXP_DATA",
    os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'exp')) + os.sep


def _build_exp_data_lib():
    if not os.path.isdir(EXP_PATH_BASE):
        raise FileNotFoundError(
            f"Experimental data directory not found: {EXP_PATH_BASE!r}. Set IDPFORGE_EXP_DATA "
            "to your Data/exp path. In a container, mount it and pass the env var, e.g. Docker "
            "`-v host/Data/exp:/data -e IDPFORGE_EXP_DATA=/data` or Apptainer "
            "`-B host/Data/exp:/data --env IDPFORGE_EXP_DATA=/data`.")
    supported = os.listdir(EXP_PATH_BASE)
    lib = {idp: {os.path.basename(f).split("_")[1]: f
                 for f in glob(EXP_PATH_BASE + idp + f"/{idp}_*_exp.txt")}
           for idp in supported}
    for idp in supported:
        d2d = EXP_PATH_BASE + idp + f"/{idp}_d2D.txt"
        if os.path.exists(d2d):
            lib[idp]["d2d"] = d2d
    return lib


class _LazyExpDataLib:
    """protein -> {data_type: exp_file}, built lazily on first access."""
    _cache = None

    def _data(self):
        if _LazyExpDataLib._cache is None:
            _LazyExpDataLib._cache = _build_exp_data_lib()
        return _LazyExpDataLib._cache

    def __getitem__(self, key):
        return self._data()[key]

    def __contains__(self, key):
        return key in self._data()

    def __iter__(self):
        return iter(self._data())

    def keys(self):
        return self._data().keys()

    def get(self, key, default=None):
        return self._data().get(key, default)


EXP_DATA_LIB = _LazyExpDataLib()


def read_bc_data(filedata, key, bc_errors=BC_ERRORS):
    if key == 'jc':
        return Stack(key, filedata,
                     {'A': np.sqrt(0.14), 'B': np.sqrt(0.03), 'C': np.sqrt(0.08)},
                     {'A': 6.51, 'B': -1.76, 'C': 1.6})
    return Stack(key, filedata, bc_errors[key], None)
