from typing import Dict, Iterable
from random import randint, choice, choices, randrange

from .utils import *


class MazeBase(ABC):
    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def connections(self):
        pass

    @abstractmethod
    def __getitem__(self, **kwargs):
        pass


class MazeGraph(MazeBase):

    @property
    def nodes(self):
        return self.__nodes

    @property
    def connections(self):
        return self.__connections

    def __init__(self, no_nodes: int, specs: list[tuple[float, CellGenerator]]):
        weights = [w for w, _ in specs]
        generators = [g for _, g in specs]

        def random_cell():
            return choices(generators, weights, k=1)[0]()

        self.__no_nodes = no_nodes

        self.__nodes: Dict[int, Cell] = \
            {
                i: random_cell()
                for i in range(self.__no_nodes)
            }

        self.__connections: Dict[int, Dict[Direction, int]] = \
            {
                node: {}
                for node in self.__nodes
                if isinstance(self.__nodes[node], RegularCell)
            }

        self.__set_maze()

    def __set_teleport(self):
        for node in self.__nodes:
            cell = self.__nodes[node]
            if isinstance(cell, TeleportCell):
                while True:
                    ti = randrange(self.__no_nodes)
                    tn = self.__nodes[ti]
                    if not isinstance(tn, WallCell) and not isinstance(tn, TeleportCell):
                        cell.state = (ti, 0)
                        cell.to_teleport_to = tn
                        break

    def __set_maze(self):

        self.__set_teleport()

        for node in self.__connections:
            directions = Direction.get_all_directions()
            no_dir = randint(1, len(directions))

            for _ in range(no_dir):
                direction = choice(directions)
                directions.remove(direction)
                to_node = choice(list(self.__nodes.keys()))
                self.__connections[node][direction] = to_node

    def __getitem__(self, key: int) -> Cell:
        return self.__nodes[key]


class MazeBoard(MazeBase):

    @property
    def nodes(self):
        return self.__nodes

    @property
    def connections(self):
        return self.__connections

    @property
    def rows_no(self):
        return self.__rows_no

    @property
    def cols_no(self):
        return self.__cols_no

    def __init__(self, size: tuple[int, int], specs: list[tuple[float, CellGenerator]]):
        width, height = size
        weights = [w for w, _ in specs]
        generators = [g for _, g in specs]

        def random_cell():
            return choices(generators, weights, k=1)[0]()

        cells = [[random_cell() for _ in range(width)] for _ in range(height)]

        self.__rows_no, self.__cols_no, cells = MazeBoard.__validate_cells(cells)

        self.__nodes: Dict[tuple[int, int], Cell] = \
            {
                (i, j): cells[i][j]
                for i in range(self.__rows_no)
                for j in range(self.cols_no)
            }

        self.__connections: Dict[tuple[int, int], Dict[Direction, tuple[int, int]]] = \
            {
                node: {}
                for node in self.__nodes
            }

        self.__set_maze()

    @staticmethod
    def __validate_cells(cells: Iterable[Iterable[Cell]]) -> tuple[int, int, list[list[Cell]]]:
        """
        Utility function used to validate the given double-iterable of cells.

        If checks are successful, it will return number of board rows and
        columns, as well as cells themselves.
        """
        cells = [list(row) for row in cells] if cells else []

        if not cells:
            raise Exception("Number of rows in a board must be at least 1.")
        if not cells[0]:
            raise Exception("There has to be at least one column.")

        rows_no = len(cells)
        cols_no = len(cells[0])

        for row in cells:
            if not row or len(row) != cols_no:
                raise Exception(
                    "Each row in a board must have the same number of columns."
                )

        return rows_no, cols_no, cells

    def __right(self, row: int, col: int):
        if col != self.__cols_no - 1:
            if self[row, col + 1].is_steppable:
                return row, col + 1
        return row, col

    def __left(self, row: int, col: int):
        if col != 0:
            if self[row, col - 1].is_steppable:
                return row, col - 1
        return row, col

    def __up(self, row: int, col: int):
        if row != 0:
            if self[row - 1, col].is_steppable:
                return row - 1, col
        return row, col

    def __down(self, row: int, col: int):
        if row != self.__rows_no - 1:
            if self[row + 1, col].is_steppable:
                return row + 1, col
        return row, col

    def __set_teleport(self):
        for node in self.__nodes:
            cell = self.__nodes[node]
            if isinstance(cell, TeleportCell):
                while True:
                    ti, tj = randrange(self.__rows_no), randrange(self.__cols_no)
                    tn = self.__nodes[(ti, tj)]
                    if not isinstance(tn, WallCell) and not isinstance(tn, TeleportCell):
                        cell.state = (ti, tj)
                        cell.to_teleport_to = tn
                        break

    def __set_maze(self):

        self.__set_teleport()

        for i, j in self.__nodes:
            self.__nodes[(i, j)].state = (i, j)
            directions = Direction.get_all_directions()

            for direction in directions:
                if direction == Direction.RIGHT:
                    di, dj = self.__right(i, j)
                elif direction == Direction.LEFT:
                    di, dj = self.__left(i, j)
                elif direction == Direction.UP:
                    di, dj = self.__up(i, j)
                elif direction == Direction.DOWN:
                    di, dj = self.__down(i, j)
                else:
                    raise Exception(
                        f"Board doesn't support {direction.name}!"
                    )

                self.__connections[(i, j)][direction] = (di, dj)

    def __getitem__(self, key: tuple[int, int]) -> Cell:
        row, col = key
        return self.__nodes[(row, col)]
