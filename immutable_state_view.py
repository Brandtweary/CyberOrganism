from typing import Dict, Any


class ImmutableStateView:
    def __init__(self, state: Dict[str, Any]):
        self._state = state

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        raise RuntimeError("Attempt to modify immutable state")

    def get(self, key, default=None):
        return self._state.get(key, default)
