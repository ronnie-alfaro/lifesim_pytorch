from enum import IntEnum


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    EAT = 4
    DRINK = 5
    REST = 6
    WAIT = 7


MOVEMENT_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
}
