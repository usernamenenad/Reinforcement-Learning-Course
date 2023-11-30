from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable


class Direction(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()

    @staticmethod
    def get_all_directions():
        return [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]


class Action(Enum):
    ACTION_A1 = auto()  # On Level 1 assignment, this would collapse to direction RIGHT
    ACTION_A2 = auto()  # On Level 1 assignment, this would collapse to direction LEFT
    ACTION_A3 = auto()  # On Level 1 assignment, this would collapse to direction UP
    ACTION_A4 = auto()  # On Level 1 assignment, this would collapse to direction DOWN

    @staticmethod
    def get_all_actions():
        return [Action.ACTION_A1, Action.ACTION_A2, Action.ACTION_A4, Action.ACTION_A3]


class Cell(ABC):
    """
    Interface class for all maze cells.
    """

    @property
    @abstractmethod
    def state(self) -> tuple[int, int]:
        pass

    @state.setter
    def state(self, state: tuple[int, int]):
        pass

    @property
    @abstractmethod
    def reward(self) -> float:
        pass

    @reward.setter
    def reward(self, reward: float):
        pass

    @property
    @abstractmethod
    def color(self) -> tuple[int, int, int]:
        pass

    @property
    def is_steppable(self) -> bool:
        return True

    @property
    def is_terminal(self) -> bool:
        return False

    @property
    def has_value(self) -> bool:
        return True


class RegularCell(Cell):
    """
    A regular cell class.

    A common, non-terminal, steppable cell.
    """

    @property
    def state(self) -> tuple[int, int]:
        return self.__state

    @state.setter
    def state(self, state: tuple[int, int]):
        self.__state = state[0], state[1]

    @property
    def reward(self) -> float:
        return self.__reward

    @reward.setter
    def reward(self, reward: float):
        self.__reward = reward

    @property
    def color(self) -> tuple[int, int, int]:
        return (255, 255, 255) if self.reward == -1 else (255, 0, 0)

    def __init__(self, reward: float):
        self.__reward: float = reward
        self.__state: tuple[int, int] = -1, -1


class TerminalCell(Cell):
    """
    A terminal cell class.

    When an agent steps onto it,
    game finishes.
    """

    @property
    def state(self) -> tuple[int, int]:
        return self.__state

    @state.setter
    def state(self, state: tuple[int, int]):
        self.__state = state[0], state[1]

    @property
    def reward(self) -> float:
        return self.__reward

    @reward.setter
    def reward(self, reward: float):
        self.__reward = reward

    @property
    def color(self) -> tuple[int, int, int]:
        return 0, 0, 255

    @property
    def is_terminal(self) -> bool:
        return True

    def __init__(self, reward: float):
        self.__reward: float = reward
        self.__state: tuple[int, int] = -1, -1


class TeleportCell(Cell):
    """
    A teleport cell class.

    When stepped onto it, agent can teleport only on
    regular or terminal cells, but not on other
    teleports nor wall cells.
    """

    @property
    def state(self) -> tuple[int, int]:
        return self.__state

    @state.setter
    def state(self, state: tuple[int, int]):
        self.__state = state[0], state[1]

    @property
    def reward(self) -> float:
        return self.__reward

    @reward.setter
    def reward(self, reward: float):
        self.__reward = reward

    @property
    def color(self) -> tuple[int, int, int]:
        return 0, 255, 0

    @property
    def to_teleport_to(self) -> Cell:
        return self.__to_teleport_to

    @to_teleport_to.setter
    def to_teleport_to(self, to_teleport_to: Cell):
        self.__to_teleport_to = to_teleport_to

    @property
    def is_steppable(self) -> bool:
        return self.to_teleport_to.is_steppable

    @property
    def is_terminal(self) -> bool:
        return self.to_teleport_to.is_terminal

    @property
    def has_value(self) -> bool:
        return self.to_teleport_to.has_value

    def __init__(self):
        self.__to_teleport_to: Cell = None
        self.__reward: float = None
        self.__state: tuple[int, int] = -1, -1


class WallCell(Cell):
    """
    A wall cell class.

    A non steppable, wall cell.
    """

    @property
    def state(self) -> tuple[int, int]:
        return self.__state

    @state.setter
    def state(self, state: tuple[int, int]):
        self.__state = state[0], state[1]

    @property
    def reward(self) -> float:
        return self.__reward

    @reward.setter
    def reward(self, reward: float):
        self.__reward = reward

    @property
    def color(self) -> tuple[int, int, int]:
        return 128, 128, 128

    @property
    def is_steppable(self) -> bool:
        return False

    @property
    def has_value(self) -> bool:
        return False

    def __init__(self):
        self.__reward: float = 0
        self.__state: tuple[int, int] = -1, -1


CellGenerator = Callable[[], Cell]
