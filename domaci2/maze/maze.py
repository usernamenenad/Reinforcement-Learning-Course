import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Iterable, Callable
from copy import copy
from random import random, choices


class Cell(ABC):
    """
    Abstract base class for all maze cells.
    """
    @abstractmethod
    def get_reward(self) -> float:
        """
        The reward an agent receives when stepping onto this cell.
        """
        pass

    def is_steppable(self) -> bool:
        """
        Checks if an agent can step onto this cell.

        Regular and terminal cells are steppable.
        Walls are not steppable.
        """
        return True

    def is_terminal(self) -> bool:
        """
        Checks if the cell is terminal.

        When stepping onto a terminal cell the agent exits
        the maze and finishes the game.
        """
        return False

    def has_value(self) -> bool:
        """
        Check if the cell has value.

        The value is defined for regular cells and terminal cells,
        but not for walls.
        """
        return True


class RegularCell(Cell):
    """
    A common, non-terminal, steppable cell.
    """
    def __init__(self, reward: float) -> None:
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class TerminalCell(Cell):
    """
    A terminal cell.
    """

    def __init__(self, reward: float) -> None:
        self.reward = reward


if __name__ == '__main__':
    print('Hi! I am maze!')
