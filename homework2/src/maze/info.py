import matplotlib.pyplot as plt
import networkx as nx
from colormap import rgb2hex
from tabulate import tabulate

from .maze import *
from .policy import *


class Info:
    @staticmethod
    def __draw_board(board: MazeBoard, ax=None):
        board_img = np.ones(shape=(board.rows_no, board.cols_no, 3), dtype=np.uint8)

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
    def __draw_board_values(env: MazeEnvironment, ax=None):
        Info.__draw_board(env.base, ax=ax)
        for s in env.states:
            ax.text(s[1] - 0.4, s[0] + 0.1, f"{env.v[s]:.1f}")

    @staticmethod
    def __draw_board_policy(env: MazeEnvironment, policy: Policy, ax=None):
        Info.__draw_board(env.base, ax=ax)
        for s in env.states:
            if not env.base[s].is_terminal:
                a = policy.act(s, env, env.actions)
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
    def __draw_graph(graph: MazeGraph, labels: dict[State, str] = None, ax=None):
        g = nx.DiGraph()
        colors = dict()
        labels = labels if labels else dict()

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
        ew = nx.get_edge_attributes(g, 'weight')
        weights = [ew[edge] for edge in ew]
        norm = plt.Normalize(min(weights), max(weights))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(weights), max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        ec = [cmap(norm(weight)) for weight in weights]

        nx.draw(g,
                pos=pos,
                labels=labels,
                edge_color=ec,
                width=2,
                font_size=10,
                with_labels=True,
                node_color=[colors[node] for node in colors],
                node_size=2000,
                edgecolors='black',
                ax=ax)

        return g, colors

    @staticmethod
    def __draw_graph_values(env: MazeEnvironment, ax=None):
        graph = env.base

        labels: dict[State, str] = dict()

        for node in graph.nodes:
            cell = graph.nodes[node]
            if node in env.states:
                labels[node] = str(len(labels)) + f", {env.v[node]:.1f}"
            else:
                if isinstance(cell, TeleportCell):
                    labels[node] = f"{len(labels)}, {env.base.find_position(cell.teleport_to)}"
                elif isinstance(cell, WallCell):
                    labels[node] = str(len(labels))

        Info.__draw_graph(graph, labels, ax)

    @staticmethod
    def __draw_graph_policy(env: MazeEnvironment, policy: Policy, ax=None):

        labels = {}

        for s in env.states:
            if not env.base[s].is_terminal:
                a = policy.act(s, env, env.actions)
                if a == Action.ACTION_A1:
                    labels[s] = 'A1'
                elif a == Action.ACTION_A2:
                    labels[s] = 'A2'
                elif a == Action.ACTION_A3:
                    labels[s] = 'A3'
                else:
                    labels[s] = 'A4'

        Info.__draw_graph(env.base, labels, ax)

    @staticmethod
    def draw_base(base: MazeBase, ax=None):
        ax = ax if ax else plt
        if isinstance(base, MazeBoard):
            Info.__draw_board(base, ax=ax)
        elif isinstance(base, MazeGraph):
            Info.__draw_graph(base, ax=ax)

    @staticmethod
    def draw_values(env: MazeEnvironment, ax=None):
        ax = ax if ax else plt
        if isinstance(env.base, MazeBoard):
            Info.__draw_board_values(env, ax)
        elif isinstance(env.base, MazeGraph):
            Info.__draw_graph_values(env, ax)

    @staticmethod
    def draw_policy(env: MazeEnvironment, policy: Policy, ax=None):
        ax = ax if ax else plt
        if isinstance(env.base, MazeBoard):
            Info.__draw_board_policy(env, policy, ax)
        elif isinstance(env.base, MazeGraph):
            Info.__draw_graph_policy(env, policy, ax)

    @staticmethod
    def print_probabilities(env: MazeEnvironment):
        to_print = list()
        for s, a in env.probabilities:
            news = env(s, a)
            for new in news:
                to_print.append(
                    {
                        "State": s.value,
                        "Action": a.name,
                        "Direction": new["Direction"].name,
                        "Next state": new["New state"].value,
                        "Reward": new["Reward"],
                        "Probability(s+, r | s, a)": env.probabilities[(s, a)][
                            new["Direction"]
                        ],
                    }
                )

        print(tabulate(to_print, "keys", "rst"))
