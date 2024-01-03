import warnings
import os.path

from alive_progress import alive_bar
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

    def update(self, *new_state):
        s: State = new_state[0][0]
        a: Action = new_state[0][1]
        r: float = new_state[0][2]
        new_s: State = new_state[0][3]

        if new_s:
            v_plus = self.q.determine_v(new_s)
        else:
            v_plus = 0

        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * v_plus)

    def run(self, game: Game, iterations: int) -> Q:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Q-Learning...")

        if os.path.exists("game_log_ql.txt"):
            os.remove("game_log_ql.txt")

        with alive_bar(iterations) as bar:
            for i in range(iterations):
                # Play a game
                game.play(EpsilonGreedyPolicy(epsilon=0.3), self.q, self.gamma)

                # Log game information in a text file
                Info.log_game(game, i, "ql")

                for player in game.players:
                    for rnd in player.experiences:
                        player.experiences[rnd].clear()

                bar()

        print("Finished Q-Learning!")
        return self.q


class SARSA(TD, Observer):

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.05):
        super().__init__(q if q else Q(), gamma, alpha)

    def update(self, *new_state):
        s: State = new_state[0][0]
        a: Action = new_state[0][1]
        r: float = new_state[0][2]
        new_s: State = new_state[0][3]
        new_a: Action = new_state[0][4]

        if new_s:
            q_plus = self.q[new_s, new_a]
        else:
            q_plus = 0.0

        self.q[s, a] = (1 - self.gamma) * self.q[s, a] + self.gamma * (r + self.gamma * q_plus)

    def run(self, game: Game, iterations: int):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting SARSA...")

        if os.path.exists("game_log_sarsa.txt"):
            os.remove("game_log_sarsa.txt")

        with alive_bar(iterations) as bar:
            for i in range(iterations):
                # Play a game
                game.play(EpsilonGreedyPolicy(epsilon=0.1), self.q, self.gamma)

                Info.log_game(game, i, "sarsa")

                for player in game.players:
                    for rnd in player.experiences:
                        player.experiences[rnd].clear()

                bar()

        print("Finished SARSA!")
        return self.q
