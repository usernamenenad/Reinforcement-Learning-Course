import matplotlib.pyplot as plt
import networkx as nx
from colormap import rgb2hex
from tabulate import tabulate

from .agent import *
from .environment import *


class Info:
    @staticmethod
    def __draw_board(board: MazeBoard, ax=None):
        board_img = np.ones(
            shape=(board.rows_no, board.cols_no, 3), dtype=np.uint8)

        for i in range(board.rows_no):
            for j in range(board.cols_no):
                board_img[i, j, :] = board[i, j].color
                if isinstance(board[i, j], TeleportCell):
                    ax.text(
                        j - 0.4,
                        i + 0.1,
                        f"({board[i, j].to_teleport_to.position[0]},"
                        f"{board[i, j].to_teleport_to.position[1]})",
                    )

        ax.imshow(board_img)

    @staticmethod
    def __draw_board_values(agent: Agent, ax=None):
        Info.__draw_board(agent.env.base, ax=ax)
        for s in agent.env.states:
            ax.text(s[1] - 0.4, s[0] + 0.1, f"{agent.env.v_values[s]:.1f}")

    @staticmethod
    def __draw_board_policy(agent: Agent, policy: str, ax=None):
        policy = GreedyPolicyV() if policy == "greedy_v" else GreedyPolicyQ()
        Info.__draw_board(agent.env.base, ax=ax)
        sa = agent.determine_optimal_actions(policy)
        for s in sa:
            if sa[s] == Action.ACTION_A1:
                ax.text(s[1] - 0.25, s[0] + 0.1, "A1")
            elif sa[s] == Action.ACTION_A2:
                ax.text(s[1] - 0.25, s[0] + 0.1, "A2")
            elif sa[s] == Action.ACTION_A3:
                ax.text(s[1] - 0.25, s[0] + 0.1, "A3")
            else:
                ax.text(s[1] - 0.25, s[0] + 0.1, "A4")

    @staticmethod
    def __draw_graph(graph: MazeGraph, ax=None):
        g = nx.DiGraph()
        colors = {}
        labels = {}

        # Defining nodes
        for node in graph.nodes:
            g.add_node(node.value)

        # Defining edges
        for node in graph.nodes:
            labels[node] = len(labels)
            cell = graph.nodes[node]
            colors[node] = rgb2hex(cell.color[0], cell.color[1], cell.color[2])
            if node in graph.connections:
                dn = graph.connections[node]
                for direction in dn:
                    to_node = graph.connections[node][direction]
                    g.add_edge(node.value, to_node.value,
                               weight=graph[to_node].reward)

        pos = nx.shell_layout(g)
        cmap = plt.cm.RdBu
        ew = nx.get_edge_attributes(g, 'weight')
        weights = [ew[edge] for edge in ew]
        norm = plt.Normalize(min(weights), max(weights))
        ec = [cmap(norm(weight)) for weight in weights]

        nx.draw(g,
                pos=pos,
                labels=labels,
                edge_color=ec,
                width=2,
                font_size=10,
                with_labels=True,
                node_color=[colors[node] for node in colors],
                node_size=1500,
                edgecolors='black',
                ax=ax)

        return g, colors

    @staticmethod
    def __draw_graph_values(agent: Agent, ax=None):
        graph = agent.env.base
        g, colors = Info.__draw_graph(graph, ax)

        labels = {}

        for node in graph.nodes:
            cell = graph.nodes[node]
            if node in agent.env.states:
                labels[node] = str(node.value) + \
                    f', {agent.env.v_values[node]:.1f}'
            else:
                if isinstance(cell, TeleportCell):
                    labels[node] = cell.to_teleport_to.position.value
                elif isinstance(cell, WallCell):
                    labels[node] = node.value

        pos = nx.shell_layout(g)
        cmap = plt.cm.RdBu
        ew = nx.get_edge_attributes(g, 'weight')
        weights = [ew[edge] for edge in ew]
        norm = plt.Normalize(min(weights), max(weights))
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

    @staticmethod
    def __draw_graph_policy(agent: Agent, policy: str, ax=None):
        policy = GreedyPolicyV() if policy == "greedy_v" else GreedyPolicyQ()
        g, colors = Info.__draw_graph(agent.env.base, ax=ax)

        labels = {}

        pos = nx.shell_layout(g)
        cmap = plt.cm.RdBu
        ew = nx.get_edge_attributes(g, 'weight')
        weights = [ew[edge] for edge in ew]
        norm = plt.Normalize(min(weights), max(weights))
        ec = [cmap(norm(weight)) for weight in weights]

        sa = agent.determine_optimal_actions(policy)
        for s in sa:
            if sa[s] == Action.ACTION_A1:
                labels[s] = 'A1'
            elif sa[s] == Action.ACTION_A2:
                labels[s] = 'A2'
            elif sa[s] == Action.ACTION_A3:
                labels[s] = 'A3'
            else:
                labels[s] = 'A4'

        nx.draw(g,
                pos=pos,
                labels=labels,
                edge_color=ec,
                width=2,
                font_size=10,
                with_labels=True,
                node_color=[colors[node] for node in colors],
                node_size=1500,
                edgecolors='black',
                ax=ax)

    @staticmethod
    def draw_base(base: MazeBase, ax=None):
        ax = ax if ax else plt
        if isinstance(base, MazeBoard):
            Info.__draw_board(base, ax)
        elif isinstance(base, MazeGraph):
            Info.__draw_graph(base, ax)

    @staticmethod
    def draw_values(agent: Agent, ax=None):
        ax = ax if ax else plt
        if isinstance(agent.env.base, MazeBoard):
            Info.__draw_board_values(agent, ax)
        elif isinstance(agent.env.base, MazeGraph):
            Info.__draw_graph_values(agent, ax)

    @staticmethod
    def draw_policy(agent: Agent, policy: str, ax=None):
        ax = ax if ax else plt
        if isinstance(agent.env.base, MazeBoard):
            Info.__draw_board_policy(agent, policy, ax)
        elif isinstance(agent.env.base, MazeGraph):
            Info.__draw_graph_policy(agent, policy, ax)

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
