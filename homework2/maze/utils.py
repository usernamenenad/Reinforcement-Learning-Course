from abc import ABC, abstractmethod
from enum import Enum, auto
from random import choices
from typing import Callable


class State:
    def __init__(self, position: list[int]):
        self.__position: list[int] = position

    def __getitem__(self, key: int):
        return self.__position[key]

    def __hash__(self):
        return hash(self.__position.__hash__)

    def __eq__(self, other):
        return (
            self.__position == other.__position
            if isinstance(other, State)
            else self.__position == list(other)
        )

    def __str__(self):
        return str(self.__position)


class Cell(ABC):
    """
    Interface class for all maze cells.
    """

    @property
    def reward(self) -> float:
        return self.__reward

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

    def __init__(self, reward: float = None):
        self.__reward = reward


class RegularCell(Cell):
    """
    A common, non-terminal, steppable cell.
    """

    @property
    def color(self) -> tuple[int, int, int]:
        return (255, 255, 255) if self.reward == -1 else (255, 0, 0)

    def __init__(self, reward: float):
        super().__init__(reward)


class TerminalCell(Cell):
    """
    A terminal cell class.

    When an agent steps onto it,
    game finishes.
    """

    @property
    def color(self) -> tuple[int, int, int]:
        return 0, 0, 255

    @property
    def is_terminal(self) -> bool:
        return True

    def __init__(self, reward: float):
        super().__init__(reward)


class TeleportCell(Cell):
    """
    A teleport cell class.

    When stepped onto it, agent can teleport only on
    regular or terminal cells, but not on other
    teleports nor wall cells.
    """

    @property
    def reward(self) -> float:
        return self.__teleport_to.reward

    @property
    def color(self) -> tuple[int, int, int]:
        return 0, 255, 0

    @property
    def teleport_to(self) -> Cell:
        return self.__teleport_to

    @teleport_to.setter
    def teleport_to(self, teleport_to: Cell):
        self.__teleport_to = teleport_to

    @property
    def is_steppable(self) -> bool:
        return self.teleport_to.is_steppable

    @property
    def is_terminal(self) -> bool:
        return self.teleport_to.is_terminal

    def __init__(self):
        super().__init__()
        self.__teleport_to = None


class WallCell(Cell):
    """
    A non steppable, wall cell.
    """

    @property
    def color(self) -> tuple[int, int, int]:
        return 128, 128, 128

    @property
    def is_steppable(self) -> bool:
        return False

    def __init__(self, reward: float):
        super().__init__(reward)


class CellGen:
    def __call__(self, specs: list[tuple[float, Callable]]) -> Cell:
        return choices(
            population=[call for _, call in specs],
            weights=[weight for weight, _ in specs],
            k=1,
        )[0]()


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
        return [Action.ACTION_A1, Action.ACTION_A2, Action.ACTION_A3, Action.ACTION_A4]


ad_map: dict[Action, Direction] = {
    Action.ACTION_A1: Direction.RIGHT,
    Action.ACTION_A2: Direction.LEFT,
    Action.ACTION_A3: Direction.UP,
    Action.ACTION_A4: Direction.DOWN,
}
