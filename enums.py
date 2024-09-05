from enum import Enum


class ObjectType(Enum):
    ITEM = 'item'
    ORGANISM = 'organism'

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NO_MOVE = 4