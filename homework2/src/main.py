from matplotlib import pyplot as plt

from maze import *
from maze.info import Info
from maze.policy import *

if __name__ == '__main__':
    DEFAULT_SPECS = [
        (10, lambda: RegularCell(-1)),
        (2, lambda: RegularCell(-10)),
        (2, lambda: WallCell(-1)),
        (1, lambda: TerminalCell(-1)),
        (1, lambda: TeleportCell())
    ]

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axes = axes.flatten()

    base = MazeBoard(size=(8, 8), specs=DEFAULT_SPECS)
    env = MazeEnvironment(base=base,
                          env_type=EnvType.DETERMINISTIC,
                          gamma=1.0)

    k = env.compute_values()
    Info.draw_values(env, ax=axes[0])
    Info.draw_policy(env, GreedyPolicyV(), ax=axes[1])
    plt.show()
