from random import randint, choice
from typing import Any

from maze.utils import *


class MazeBase(ABC):
    """
    Base class for maze.
    """

    @property
    def nodes(self) -> dict[State, Cell]:
        return self.__nodes

    @property
    def connections(self) -> dict[State, dict[Direction, State]]:
        return self.__connections

    def __init__(self, positions: list[list[int]], specs: list[tuple[float, Callable]]):
        self.__nodes: dict[State, Cell] = {
            State(position): CellGen()(specs) for position in positions
        }

        self.__connections: dict[State, dict[Direction, State]] = {
            node: {} for node in self.__nodes
        }

    def __getitem__(self, state: Any) -> Cell:
        return self.__nodes[state if isinstance(state, State) else State(list(state))]

    def __iter__(self):
        return iter(self.__nodes)

    def set_teleport(self):
        """
        Private method for configuring teleport cells - to what
        cells will agent teleport when stepped onto teleport cell.
        """

        valid_teleports = [
            cell
            for cell in self.__nodes.values()
            if not isinstance(cell, WallCell) and not isinstance(cell, TeleportCell)
        ]

        for node in self.__nodes:
            cell = self.__nodes[node]
            if isinstance(cell, TeleportCell):
                cell.teleport_to = choice(valid_teleports)

    def find_position(self, cell: Cell) -> State:
        return list(self.__nodes.keys())[list(self.__nodes.values()).index(cell)]

    def get_directions(self, s: State) -> list[Direction]:
        return list(self.__connections[s].keys())

    def get_from(self, node: State, direction: Direction) -> State:
        return self.__connections[node][direction]

    @abstractmethod
    def set_maze(self) -> None:
        pass


class MazeGraph(MazeBase):
    """
    Inherited from MazeBase class - it models a graph.
    """

    def __init__(self, size: int, specs: list[tuple[float, Callable]]):
        super().__init__(positions=[[i] for i in range(size)], specs=specs)

        self.set_maze()

    def set_maze(self):
        """
        Private method for creating graphs -
        making nodes and random edges.
        """

        self.set_teleport()

        # # ALL FOUR DIRECTIONS USED PER NODE
        # for node in self.connections:
        #     if isinstance(self[node], RegularCell):
        #         for direction in Direction.get_all_directions():
        #             self.connections[node][direction] = choice(list(self.nodes.keys()))

        # CUSTOM NUMBER OF DIRECTIONS USED PER NODE
        for node in self.connections:
            if isinstance(self[node], RegularCell):
                directions = Direction.get_all_directions()
                no_dir = randint(1, len(directions))

                for _ in range(no_dir):
                    direction = choice(directions)
                    directions.remove(direction)
                    self.connections[node][direction] = choice(list(self.nodes.keys()))


class MazeBoard(MazeBase):
    """
    Inherited from MazeBase class - models a board.
    """

    @property
    def rows_no(self) -> int:
        return self.__rows_no

    @property
    def cols_no(self) -> int:
        return self.__cols_no

    @property
    def size(self) -> tuple[int, int]:
        return self.__rows_no, self.__cols_no

    def __init__(self, size: tuple[int, int], specs: list[tuple[float, Callable]]):
        self.__rows_no, self.__cols_no = size

        super().__init__(positions=[[i, j] for i in range(size[0]) for j in range(size[1])],
                         specs=specs)

        self.set_maze()

    def __right(self, position: State) -> State:
        row, col = position[0], position[1]
        if col != self.__cols_no - 1:
            if self[row, col + 1].is_steppable:
                return State([row, col + 1])
        return State([row, col])

    def __left(self, position: State) -> State:
        row, col = position[0], position[1]
        if col != 0:
            if self[row, col - 1].is_steppable:
                return State([row, col - 1])
        return State([row, col])

    def __up(self, position: State) -> State:
        row, col = position[0], position[1]
        if row != 0:
            if self[row - 1, col].is_steppable:
                return State([row - 1, col])
        return State([row, col])

    def __down(self, position: State) -> State:
        row, col = position[0], position[1]
        if row != self.__rows_no - 1:
            if self[row + 1, col].is_steppable:
                return State([row + 1, col])
        return State([row, col])

    def set_maze(self):
        """
        Private method for creating board -
        making board squares (here named nodes).
        """

        self.set_teleport()

        for node in self.nodes:
            for direction in Direction.get_all_directions():
                match direction:
                    case Direction.RIGHT:
                        dc = self.__right(node)
                    case Direction.LEFT:
                        dc = self.__left(node)
                    case Direction.UP:
                        dc = self.__up(node)
                    case Direction.DOWN:
                        dc = self.__down(node)
                    case _:
                        raise Exception(
                            f"No direction {direction} supported for this type of maze!"
                        )

                for dnode in self.nodes:
                    if dnode == dc:
                        self.connections[node][direction] = dnode
