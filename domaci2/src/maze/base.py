from random import randint, choice, choices
from typing import Dict, Iterable

from .utils import *


class MazeBase(ABC):
    """
    Base class for maze.
    """

    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def connections(self):
        pass

    @abstractmethod
    def __getitem__(self, **kwargs) -> Cell:
        pass

    @abstractmethod
    def get_directions(self, s: Position) -> list[Direction]:
        pass


class GraphPosition(Position):
    """
    Inherited from Position class - models a graph node.
    """

    @property
    def value(self) -> int:
        return self.__value

    def __init__(self, value: int):
        self.__value = value

    def __eq__(self, other):
        if isinstance(other, int):
            return self.__value == other
        elif isinstance(other, GraphPosition):
            return self.__value == other.__value

        raise Exception(
            f"Cannot use '==' in context of {other.__class__.__name__}"
        )

    def __hash__(self):
        return hash(self.__value)


class MazeGraph(MazeBase):
    """
    Inherited from MazeBase class - it models a graph.
    """

    @property
    def nodes(self) -> Dict[Position, Cell]:
        return self.__nodes

    @property
    def connections(self) -> Dict[Position, Dict[Direction, Position]]:
        return self.__connections

    @property
    def size(self) -> int:
        return self.__no_nodes

    def __init__(self, no_nodes: int, specs: list[tuple[float, CellGenerator]]):
        weights = [w for w, _ in specs]
        generators = [g for _, g in specs]

        def random_cell():
            return choices(generators, weights, k=1)[0]()

        self.__no_nodes = no_nodes

        self.__nodes: Dict[Position, Cell] = \
            {
                GraphPosition(i): random_cell()
                for i in range(self.__no_nodes)
        }

        self.__connections: Dict[Position, Dict[Direction, Position]] = \
            {
                node: {}
                for node in self.__nodes
        }

        self.__set_maze()

    def __call__(self, position: int) -> Position:
        for node in self.__nodes:
            if node == position:
                return node

        raise Exception(
            f"No position {position} in this base!"
        )

    def __getitem__(self, key: int | Position) -> Cell:
        for node in self.__nodes:
            if node == key:
                return self.__nodes[node]

        raise Exception(
            f"No item {key} in this base!"
        )

    def __set_teleport(self):
        """
        Private method for configuring teleport cells - to what
        cells will agent teleport when stepped onto teleport cell.
        """
        for node in self.__nodes:
            self.__nodes[node].position = node
            cell = self.__nodes[node]
            if isinstance(cell, TeleportCell):
                while True:
                    tp = choice(list(self.__nodes.keys()))
                    tn = self.__nodes[tp]
                    if not isinstance(tn, WallCell) and not isinstance(tn, TeleportCell):
                        cell.position = tp
                        cell.to_teleport_to = tn
                        break

    def __set_maze(self):
        """
        Private method for creating graphs -
        making nodes and random edges.
        """

        self.__set_teleport()

        for node in self.__connections:
            if isinstance(self[node], RegularCell):
                directions = Direction.get_all_directions()
                no_dir = randint(1, len(directions))

                for _ in range(no_dir):
                    direction = choice(directions)
                    directions.remove(direction)
                    to_node = choice(list(self.__nodes.keys()))
                    self.__connections[node][direction] = to_node

    def compute_direction(self, node: Position, direction: Direction) -> Position:
        """
        Returns a node that is in direction from a given node. If that is a wall cell,
        it returns given node.
        """
        to_node = self.__connections[node][direction]
        cell = self.__nodes[to_node]
        if isinstance(cell, WallCell):
            return node
        return to_node

    def get_directions(self, node: Position) -> list[Direction]:
        return list(self.__connections[node].keys())


class BoardPosition(Position):
    """
    Inherited from Position class - models a board square.
    """

    @property
    def value(self) -> tuple[int, int]:
        return self.__value

    def __init__(self, value: tuple[int, int]):
        self.__value = value

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.__value == other
        elif isinstance(other, BoardPosition):
            return self.__value == other.__value

        raise Exception(
            f"Cannot use '==' in context of {other.__class__.__name__}"
        )

    def __hash__(self):
        return hash(self.__value)

    def __getitem__(self, key: int):
        if key == 0 or key == 1:
            return self.__value[key]

        raise Exception(
            f"No position {key} in this base!"
        )


class MazeBoard(MazeBase):
    """
    Inherited from MazeBase class - models a board.
    """

    @property
    def nodes(self) -> Dict[Position, Cell]:
        return self.__nodes

    @property
    def connections(self) -> Dict[Position, Dict[Direction, Position]]:
        return self.__connections

    @property
    def rows_no(self) -> int:
        return self.__rows_no

    @property
    def cols_no(self) -> int:
        return self.__cols_no

    @property
    def size(self) -> tuple[int, int]:
        return self.__rows_no, self.__cols_no

    def __init__(self, size: tuple[int, int], specs: list[tuple[float, CellGenerator]]):
        width, height = size
        weights = [w for w, _ in specs]
        generators = [g for _, g in specs]

        def random_cell():
            return choices(generators, weights, k=1)[0]()

        cells = [[random_cell() for _ in range(width)] for _ in range(height)]

        self.__rows_no, self.__cols_no, cells = MazeBoard.__validate_cells(
            cells)

        self.__nodes: Dict[Position, Cell] = \
            {
                BoardPosition((i, j)): cells[i][j]
                for i in range(self.__rows_no)
                for j in range(self.cols_no)
        }

        self.__connections: Dict[Position, Dict[Direction, Position]] = \
            {
                node: {}
                for node in self.__nodes
        }

        self.__set_maze()

    def __call__(self, position: tuple[int, int]) -> Position:
        for node in self.__nodes:
            if node == position:
                return node

        raise Exception(
            f"No position {position} in this base!"
        )

    def __getitem__(self, key: tuple[int, int]) -> Cell:
        for node in self.__nodes:
            if node == key:
                return self.__nodes[node]

        raise Exception(
            f"No item {key} in this base!"
        )

    @staticmethod
    def __validate_cells(cells: Iterable[Iterable[Cell]]) -> tuple[int, int, list[list[Cell]]]:
        """
        Utility function used to validate the given double-iterable of cells.

        If checks are successful, it will return number of board rows and
        columns, as well as cells themselves.
        """
        cells = [list(row) for row in cells] if cells else list()

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

    def __right(self, position: tuple[int, int]):
        row, col = position
        if col != self.__cols_no - 1:
            if self[row, col + 1].is_steppable:
                return row, col + 1
        return row, col

    def __left(self, position: tuple[int, int]):
        row, col = position
        if col != 0:
            if self[row, col - 1].is_steppable:
                return row, col - 1
        return row, col

    def __up(self, position: tuple[int, int]):
        row, col = position
        if row != 0:
            if self[row - 1, col].is_steppable:
                return row - 1, col
        return row, col

    def __down(self, position: tuple[int, int]):
        row, col = position
        if row != self.__rows_no - 1:
            if self[row + 1, col].is_steppable:
                return row + 1, col
        return row, col

    def __set_teleport(self):
        """
        Private method for configuring teleport cells - to what
        cells will agent teleport when stepped onto teleport cell.
        """

        for node in self.__nodes:
            cell = self.__nodes[node]
            if isinstance(cell, TeleportCell):
                while True:
                    tp = choice(list(self.__nodes.keys()))
                    tn = self.__nodes[tp]
                    if not isinstance(tn, WallCell) and not isinstance(tn, TeleportCell):
                        cell.position = tp
                        cell.to_teleport_to = tn
                        break

    def __set_maze(self):
        """
        Private method for creating board -
        making board squares (here named nodes).
        """

        self.__set_teleport()

        for node in self.__nodes:
            self.__nodes[node].position = node
            directions = Direction.get_all_directions()

            for direction in directions:
                if direction == Direction.RIGHT:
                    di, dj = self.__right(node.value)
                elif direction == Direction.LEFT:
                    di, dj = self.__left(node.value)
                elif direction == Direction.UP:
                    di, dj = self.__up(node.value)
                elif direction == Direction.DOWN:
                    di, dj = self.__down(node.value)
                else:
                    raise Exception(
                        f"Board doesn't support {direction.name}!"
                    )
                for dnode in self.__nodes:
                    if dnode == (di, dj):
                        self.__connections[node][direction] = dnode

    def compute_direction(self, node: Position, direction: Direction) -> Position:
        """
        Returns a node that is in direction from a given node. If that is a wall cell,
        it returns given node.
        """
        return self.__connections[node][direction]

    def get_directions(self, node: Position) -> list[Direction]:
        return list(self.__connections[node].keys())
