import threading
import numpy as np
from dataclasses import dataclass, field

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

@dataclass
class TeleopAction(AttrDict):
    left: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(1)])
    right: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(1)])
    base: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # torso: float = field(default_factory=lambda: 0.)
    # extra: dict = field(default_factory=dict)


@dataclass
class TeleopObservation(AttrDict):
    left: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(2)])
    right: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(2)])
    base: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torso: float = field(default_factory=lambda: 0.)
    extra: dict = field(default_factory=dict)

