import os

import matplotlib.pyplot as plt
import networkx as nx
from colormap import rgb2hex
from numpy import ones, uint8
from tabulate import tabulate

from maze.base import MazeGraph, MazeBoard, MazeBase
from maze.env import MazeEnvironment
from maze.func import V, Q
from maze.policy import Policy
from maze.utils import *


class Info:
    @staticmethod
    def __draw_board(board: MazeBoard, ax):
        board_img = ones(shape=(board.rows_no, board.cols_no, 3), dtype=uint8)

        for i in range(board.rows_no):
            for j in range(board.cols_no):
                board_img[i, j, :] = board[i, j].color
                if isinstance(board[i, j], TeleportCell):
                    ax.text(
                        j - 0.4,
                        i + 0.1,
                        f"({board.find_position(board[i, j].teleport_to)[0]},"
                        f"{board.find_position(board[i, j].teleport_to)[1]})",
                    )

        ax.imshow(board_img)

    @staticmethod
    def __draw_board_values(env: MazeEnvironment, values: dict[State, float], ax):
        Info.__draw_board(env.base, ax=ax)
        for s in env.states:
            ax.text(s[1] - 0.4, s[0] + 0.1, f"{values[s]:.1f}")

    @staticmethod
    def __draw_board_policy(
        env: MazeEnvironment,
        values: V | Q,
        policy: Policy,
        gamma: float,
        ax,
    ):
        Info.__draw_board(env.base, ax=ax)
        for s in env.states:
            if not env.base[s].is_terminal:
                a = policy.act(s, env, values, gamma)
                match env.type:
                    case EnvType.STOCHASTIC:
                        if a == Action.ACTION_A1:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "A1")
                        elif a == Action.ACTION_A2:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "A2")
                        elif a == Action.ACTION_A3:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "A3")
                        else:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "A4")
                    case EnvType.DETERMINISTIC:
                        if ad_map[a] == Direction.RIGHT:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "→")
                        elif ad_map[a] == Direction.LEFT:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "←")
                        elif ad_map[a] == Direction.UP:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "↑")
                        else:
                            ax.text(s[1] - 0.25, s[0] + 0.1, "↓")

    @staticmethod
    def __draw_graph(graph: MazeGraph, ax, labels: dict[State, str] | None = None):
        g = nx.DiGraph()
        colors = dict()
        labels = labels if labels else {}

        # Defining nodes
        for node in graph.nodes:
            g.add_node(node)

        # Defining edges
        for node in graph.nodes:
            cell = graph.nodes[node]
            colors[node] = rgb2hex(cell.color[0], cell.color[1], cell.color[2])
            for direction in graph.get_directions(node):
                to_node = graph.connections[node][direction]
                g.add_edge(node, to_node, weight=graph[to_node].reward)

        pos = nx.shell_layout(g)
        cmap = plt.cm.RdBu
        ew = nx.get_edge_attributes(g, "weight")
        weights = [ew[edge] for edge in ew]
        norm = plt.Normalize(min(weights), max(weights))
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(min(weights), max(weights))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        ec = [cmap(norm(weight)) for weight in weights]

        nx.draw(
            g,
            pos=pos,
            labels=labels,
            edge_color=ec,
            width=1,
            font_size=6,
            with_labels=True,
            node_color=[colors[node] for node in colors],
            node_size=750,
            edgecolors="black",
            ax=ax,
        )

        return g, colors

    @staticmethod
    def __draw_graph_values(env: MazeEnvironment, vf: dict[State, float], ax):
        graph = env.base
        labels: dict[State, str] = dict()

        for node in graph.nodes:
            cell = graph.nodes[node]
            if node in env.states:
                labels[node] = str(len(labels)) + f", {vf[node]:.1f}"
            else:
                if isinstance(cell, TeleportCell):
                    labels[
                        node
                    ] = f"{len(labels)}, {env.base.find_position(cell.teleport_to)}"
                elif isinstance(cell, WallCell):
                    labels[node] = str(len(labels))

        Info.__draw_graph(graph, ax, labels=labels)

    @staticmethod
    def __draw_graph_policy(
        env: MazeEnvironment,
        vf: V | Q,
        policy: Policy,
        gamma: float,
        ax,
    ):
        labels = {}

        for s in env.states:
            if not env.base[s].is_terminal:
                a = policy.act(s, env, vf, gamma)
                if a == Action.ACTION_A1:
                    labels[s] = "A1"
                elif a == Action.ACTION_A2:
                    labels[s] = "A2"
                elif a == Action.ACTION_A3:
                    labels[s] = "A3"
                else:
                    labels[s] = "A4"

        Info.__draw_graph(env.base, ax, labels=labels)

    @staticmethod
    def draw_base(base: MazeBase, ax=None):
        ax = ax if ax else plt
        if isinstance(base, MazeBoard):
            Info.__draw_board(base, ax=ax)
        elif isinstance(base, MazeGraph):
            Info.__draw_graph(base, ax=ax)

    @staticmethod
    def draw_values(env: MazeEnvironment, vf: V | Q, ax=None):
        ax = ax if ax else plt
        v: dict[State, float] = vf.v_table
        if isinstance(env.base, MazeBoard):
            Info.__draw_board_values(env, v, ax)
        elif isinstance(env.base, MazeGraph):
            Info.__draw_graph_values(env, v, ax)

    @staticmethod
    def draw_policy(
        env: MazeEnvironment,
        vf: V | Q,
        policy: Policy,
        gamma: float,
        ax=None,
    ):
        ax = ax if ax else plt
        if isinstance(env.base, MazeBoard):
            Info.__draw_board_policy(env, vf, policy, gamma, ax)
        elif isinstance(env.base, MazeGraph):
            Info.__draw_graph_policy(env, vf, policy, gamma, ax)

    @staticmethod
    def log_probabilities(env: MazeEnvironment, nof: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        to_log = list()
        for s, a in env.probabilities:
            mdp = env(s, a)
            for value in mdp:
                to_log.append(
                    {
                        "State": s,
                        "Action": a,
                        "Direction": value["direction"],
                        "Next state": value["next_state"],
                        "Reward": value["reward"],
                        "Probability(s+, r | s, a)": env.probabilities[s, a][
                            value["direction"]
                        ],
                    }
                )

        with open(f"./logs/probabilities_{nof}.txt", "w") as p:
            p.write(tabulate(to_log, "keys", "rst"))

    @staticmethod
    def log_values(vf: V | Q, nof: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        with open(f"./logs/{vf.__class__.__name__.lower()}_values_{nof}.txt", "w") as v:
            v.write(vf.__str__())
