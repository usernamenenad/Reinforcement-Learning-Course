import warnings
import os.path

from tqdm import trange
from observer import Observer

from .info import *


class TD(ABC):

    @abstractmethod
    def __init__(self, q: Q, gamma: float, alpha: float):
        self.q = q
        self.gamma = gamma
        self.alpha = alpha

    @abstractmethod
    def run(self, game: Game, iterations: int):
        pass


class QLearning(TD, Observer):
    """
    An off-policy method.
    """

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.1):
        super().__init__(q if q else Q(), gamma, alpha)

    def run(self, game: Game, iterations: int) -> Q:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Q-Learning...")

        if os.path.exists("game_log_ql.txt"):
            os.remove("game_log_ql.txt")

        for i in trange(iterations):
            # Play a game
            game.play(EpsilonGreedyPolicy(epsilon=0.3), self.q, self.gamma)

            # Log game information in a text file
            Info.log_game(game, i, "ql")

            for player in game.players:
                for rnd in player.experiences:
                    for j, part in enumerate(player.experiences[rnd].experience):
                        s, a = part[0], part[1]
                        if j + 1 < len(player.experiences[rnd].experience):
                            r = 0.0
                            v_plus = self.q.determine_v(player.experiences[rnd][j + 1][0])
                        else:
                            r = part[2]
                            v_plus = 0.0

                        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * v_plus)

                    # This DOES NOT mean that we're going to forget experiences.
                    # All experiences of this game are
                    # transferred to `occurrences` dictionary, and we
                    # clear experiences for the next game.
                    player.experiences[rnd].clear()

        print("Finished Q-Learning!")
        return self.q


class SARSA(TD):
    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.05):
        super().__init__(q if q else Q(), gamma, alpha)

    def run(self, game: Game, iterations: int):
        pass
