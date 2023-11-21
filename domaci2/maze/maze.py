from draw import *
from agent import *


if __name__ == '__main__':
    DEFAULT_SPECS = [
        (10, lambda: RegularCell(-1)),
        (2, lambda: RegularCell(-10)),
        (2, lambda: WallCell()),
        (1, lambda: TerminalCell(-1)),
        (0.5, lambda: TeleportCell())
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axes = axes.flatten()

    board = MazeBoard(size=(8, 8), specs=DEFAULT_SPECS)
    env = MazeEnvironment(board)
    agent = Agent(env, Action.get_all_actions(), state=(0, 0))
    agent.set_policy(GreedyPolicy())
    # print(agent.get_policy())

    Draw.draw_values(agent, ax=axes[0])
    axes[0].set_title('Initial Q values.')

    k = env.compute_values()

    Draw.draw_values(agent, ax=axes[1])
    axes[1].set_title(f'V values computed using Q values after {k} iterations.')

    Draw.draw_policy(agent, 'v', ax=axes[2])
    axes[2].set_title('Optimal policy using V values')

    Draw.draw_policy(agent, 'q', ax=axes[3])
    axes[3].set_title('Optimal policy using Q values')

    plt.show()
