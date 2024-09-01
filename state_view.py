from typing import Dict, Any


class StateView:
    def __init__(self, state: Dict[str, Any], mutable: bool = False):
        self._state = state
        self._mutable = mutable

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        if not self._mutable:
            raise RuntimeError("Attempt to modify immutable state")
        self._state[key] = value

    def get(self, key, default=None):
        return self._state.get(key, default)

    def set_mutable(self, mutable: bool):
        self._mutable = mutable

    @property
    def is_mutable(self):
        return self._mutable