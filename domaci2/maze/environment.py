from abc import ABC, abstractmethod
from copy import deepcopy
from random import choices, randint
from random import random
from typing import Iterable, Callable

from actions import *


class Cell(ABC):
    """
    Abstract base class for all maze cells.
    """

    @abstractmethod
    def get_position(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass

    @abstractmethod
    def get_color(self) -> tuple[int, int, int]:
        pass

    def is_steppable(self) -> bool:
        return True

    def is_terminal(self) -> bool:
        return False

    def has_value(self) -> bool:
        return True


class RegularCell(Cell):
    """
    A regular cell class.

    A common, non-terminal, steppable cell.
    """

    def __init__(self, reward):
        self.reward: float = reward
        self.position: tuple[int, int] = None

    def get_position(self) -> tuple[int, int]:
        return self.position

    def get_reward(self) -> float:
        return self.reward

    def get_color(self) -> tuple[int, int, int]:
        return (255, 255, 255) if self.reward == -1 else (255, 0, 0)


class TerminalCell(Cell):
    """
    A terminal cell class.

    When an agent steps onto it,
    game finishes.
    """

    def __init__(self, reward):
        self.reward: float = reward
        self.position: tuple[int, int] = None

    def get_position(self) -> tuple[int, int]:
        return self.position

    def get_reward(self) -> float:
        return self.reward

    def get_color(self) -> tuple[int, int, int]:
        return 0, 0, 255

    def is_terminal(self) -> bool:
        return True


class TeleportCell(Cell):
    """
    A teleport cell class.

    When stepped onto it, agent can teleport only on
    regular or terminal cells, but not on other
    teleports nor wall cells.
    """

    def __init__(self):
        self.to_teleport_to: Cell = None
        self.reward: float = None
        self.position: tuple[int, int] = None

    def get_position(self) -> tuple[int, int]:
        return self.position

    def get_reward(self) -> float:
        return self.reward

    def get_color(self) -> tuple[int, int, int]:
        return 0, 255, 0

    def is_steppable(self) -> bool:
        return self.to_teleport_to.is_steppable()

    def is_terminal(self) -> bool:
        return self.to_teleport_to.is_terminal()

    def has_value(self) -> bool:
        return self.to_teleport_to.has_value()


class WallCell(Cell):
    """
    A wall cell class.

    A non steppable, wall cell.
    """

    def __init__(self):
        self.reward: float = 0
        self.position: tuple[int, int] = None

    def get_position(self) -> tuple[int, int]:
        return self.position

    def get_reward(self) -> float:
        return self.reward

    def get_color(self) -> tuple[int, int, int]:
        return 0, 0, 0

    def is_steppable(self) -> bool:
        return False

    def has_value(self) -> bool:
        return False


CellGenerator = Callable[[], Cell]


class MazeBoard:
    """
    A rectangular grid of cells representing a single maze.
    """

    def __init__(self, size: tuple[int, int], specs: list[tuple[float, CellGenerator]]):
        """
        Initializer for the maze board from given `cells`. This method will
        create a random board based on provided size and specs.
        """

        width, height = size

        weights = [w for w, _ in specs]
        generators = [g for _, g in specs]

        def random_cell():
            return choices(generators, weights, k=1)[0]()

        cells = [[random_cell() for _ in range(width)] for _ in range(height)]

        self.rows_no, self.cols_no, self.cells = MazeBoard.validate_cells(cells)

        self.set_cells_position()

    def __getitem__(self, key: tuple[int, int]):
        row, col = key
        return self.cells[row][col]

    @staticmethod
    def validate_cells(cells: Iterable[Iterable[Cell]]) -> tuple[int, int, list[list[Cell]]]:
        """
        Utility function used to validate the given double-iterable of cells.

        If checks are successful, it will return number of board rows and
        columns, as well as cells themselves.
        """
        cells = [list(row) for row in cells] if cells else []

        if not cells:
            raise Exception('Number of rows in a board must be at least 1.')
        if not cells[0]:
            raise Exception('There has to be at least one column.')

        rows_no = len(cells)
        cols_no = len(cells[0])

        for row in cells:
            if not row or len(row) != cols_no:
                raise Exception('Each row in a board must have the same number of columns.')

        return rows_no, cols_no, cells

    def set_cells_position(self):
        """
        A method for determining cells' position.
        Besides determining positions, it will assign
        one random cell to teleport cell that is not
        another teleport nor wall cell.
        """
        for row in range(self.rows_no):
            for col in range(self.cols_no):
                cell = self[row, col]
                cell.position = row, col
                if isinstance(cell, TeleportCell):
                    while True:
                        i, j = randint(0, self.rows_no - 1), randint(0, self.cols_no - 1)
                        if not isinstance(self[i, j], TeleportCell) and not isinstance(self[i, j], WallCell):
                            cell.to_teleport_to = self[i, j]
                            break


class MazeEnvironment:
    """
    Wrapper for a maze board that behaves like an MDP environment.

    This is a callable object that behaves like a deterministic MDP
    state transition function - given the current state and action,
    it returns the following state and reward.

    In addition, the environment object is capable of enumerating all
    possible states and all possible actions, as well as determining
    if the state is terminal.
    """

    def __init__(self, board: MazeBoard, gamma: float = 1):
        """
        Initializer for the environment by specifying the underlying
        maze board.
        """
        self.board = board
        self.states = [(i, j) for i in range(self.board.rows_no) for j in range(self.board.cols_no)
                       if self.board[i, j].is_steppable() and not isinstance(self.board[i, j], TeleportCell)]
        self.q_values = {(s, a): -10 * random() if not self.is_terminal(s) else 0
                         for s in self.states for a in self.get_actions()}
        self.gamma = gamma

    def __call__(self, state: tuple[int, int], action: Action):
        row, col = state

        new_row, new_col = self.compute_action(row, col, action)
        new_cell = self.board[new_row, new_col]
        if isinstance(new_cell, TeleportCell):
            new_row = new_cell.to_teleport_to.get_position()[0]
            new_col = new_cell.to_teleport_to.get_position()[1]
            new_cell = new_cell.to_teleport_to
        reward = new_cell.get_reward()
        is_terminal = new_cell.is_terminal()

        return (new_row, new_col), reward, is_terminal

    def validate_position(self, row: int, col: int):
        """
        A utility function that validates a position.
        """
        if row < 0 or row >= self.board.rows_no:
            raise Exception()
        if col < 0 or col >= self.board.cols_no:
            raise Exception()
        if not self.board[row, col].is_steppable():
            raise Exception()

    def compute_action(self, row: int, col: int, a: Action) -> tuple[int, int]:
        """
        Compute action for a certain environment. Firstly, we define inner functions for movement in all
        4 directions. After, we define the move function itself.
        """

        if a not in self.get_actions():
            raise Exception(f'Agent cannot take action {a.name} in this environment.')

        def right(board: MazeBoard, row: int, col: int) -> tuple[int, int]:
            if col != board.cols_no - 1:
                if board[row, col + 1].is_steppable():
                    return row, col + 1
            return row, col

        def left(board: MazeBoard, row: int, col: int) -> tuple[int, int]:
            if col != 0:
                if board[row, col - 1].is_steppable():
                    return row, col - 1
            return row, col

        def up(board: MazeBoard, row: int, col: int) -> tuple[int, int]:
            if row != 0:
                if board[row - 1, col].is_steppable():
                    return row - 1, col
            return row, col

        def down(board: MazeBoard, row: int, col: int) -> tuple[int, int]:
            if row != board.rows_no - 1:
                if board[row + 1, col].is_steppable():
                    return row + 1, col
            return row, col

        if a == Action.RIGHT:
            return right(self.board, row, col)
        elif a == Action.LEFT:
            return left(self.board, row, col)
        elif a == Action.UP:
            return up(self.board, row, col)
        else:
            return down(self.board, row, col)

    def determine_v(self, s: tuple[int, int]):
        q = []
        for a in self.get_actions():
            q.append(self.q_values[(s, a)])

        return max(q)

    def update_q_values(self):
        for s in self.states:
            if not self.is_terminal(s):
                for a in self.get_actions():
                    s_new, r, _ = self(s, a)
                    self.q_values[(s, a)] = r + self.gamma * self.determine_v(s_new)

    def compute_q_values(self, eps: float = 0.01, max_iter: int = 1000):
        for k in range(max_iter):
            ov = deepcopy(self.q_values)
            self.update_q_values()
            err = max([abs(self.q_values[(s, a)] - ov[(s, a)]) for s, a in self.q_values])
            if err < eps:
                return k

        return max_iter

    def greedy_policy(self, s):
        v = []
        for a in self.get_actions():
            s_new, r, _ = self(s, a)
            v.append((self.determine_v(s_new), s, a))

        return max(v, key=lambda x: x[0])

    def get_actions(self):
        """
        Returning actions that are possible to take in this
        environment.
        """
        return Action.get_all_actions()

    def is_terminal(self, state: tuple[int, int]):
        return self.board[state].is_terminal()


if __name__ == '__main__':
    print('Hi! Here you can find implementation of #Cell, '
          'MazeBoard and MazeEnvironment class, used for constructing '
          'agent environment')
