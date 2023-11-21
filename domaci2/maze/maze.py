from agent import *
from draw import *

if __name__ == '__main__':
    DEFAULT_SPECS = [
        (10, lambda: RegularCell(-1)),
        (2, lambda: RegularCell(-10)),
        (2, lambda: WallCell()),
        (1, lambda: TerminalCell(-1)),
        (0.5, lambda: TeleportCell())
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
    axes = axes.flatten()

    board = MazeBoard(size=(8, 8), specs=DEFAULT_SPECS)
    env = MazeEnvironment(board)
    agent = Agent(env, (0, 0), Action.get_all_actions())

    Draw.draw_values(env, ax=axes[0])
    axes[0].set_title('Initial Q values.')

    k = env.compute_q_values()

    Draw.draw_values(env, ax=axes[1])
    axes[1].set_title(f'V values computed using Q values after {k} iterations.')

    Draw.draw_policy(env, ax=axes[2])
    axes[2].set_title('Optimal policy')

    plt.show()
