import unittest

from bandit import *


class TestMultiarmBandits(unittest.TestCase):
    """
    A class for testing multiarm bandit
    """

    def test_single_bandit(self) -> None:
        """
        Testing a single bandit.
        """
        test_mean = 1.0
        test_span = 3.0
        test_len = 1000

        test_bandit = Bandit(test_mean, test_span)
        test_rewards = [test_bandit.pull_leaver() for _ in range(test_len)]

        plt.plot(test_rewards, label='rewards')
        plt.plot((test_mean + test_span) * np.ones(test_len),
                 linestyle='--', color='red')
        plt.plot((test_mean - test_span) * np.ones(test_len),
                 linestyle='--', color='red')

        plt.show()

    def test_environment(self) -> None:
        """
        Testing bandits' environment. 
        """
        test_env_size = 5
        test_bandits = [Bandit(i ** 2, i) for i in range(test_env_size)]
        test_env = BanditsEnvironment(test_bandits)
        test_len = 1000

        selected_bandit = 4
        test_rewards = [test_env.take_action(
            selected_bandit) for _ in range(test_len)]

        plt.plot(test_rewards, label='rewards')
        plt.plot((selected_bandit ** 2 + selected_bandit) *
                 np.ones(test_len), linestyle='--', color='r')
        plt.plot((selected_bandit ** 2 - selected_bandit) *
                 np.ones(test_len), linestyle='--', color='r')

        plt.show()

        test_rewards = [test_env.take_action(
            random.randint(0, 4)) for _ in range(test_len)]
        test_mean = sum(test_rewards) / test_len

        print("TEST MEAN = ", test_mean)

    def test_decision_policies(self) -> None:
        """
        Testing decision policies.
        """
        test_q = [1, 2, 3, 2, 1]
        test_len = 1000

        plt.subplot(3, 1, 1)
        plt.plot([GreedyPolicy.action(q=test_q) for _ in range(test_len)])
        plt.subplot(3, 1, 2)
        plt.plot([RandomPolicy.action(q=test_q) for _ in range(test_len)])
        plt.subplot(3, 1, 3)
        plt.plot([EpsGreedyPolicy.action(q=test_q, eps=0.1)
                 for _ in range(test_len)])

        plt.show()

    def test_system(self) -> None:

        BANDITS_NO = 5
        ATTEMPTS_NO = 10000

        bandits = [Bandit(10 * (random.random() - 0.5), 5 *
                          random.random()) for _ in range(BANDITS_NO)]
        sys = System(bandits)

        #  *** 1. zadatak ***
        # Razlog manjeg nagiba jeste što smanjivanjem epsilon vrijednosti smanjujemo eksploraciju i držimo se
        # eksploatacije, te u tom slučaju kriva će biti bliža "optimalnoj".
        # Prikazujemo i grafik konvergencije, koji pokazuje suštinu epsilon greedy politike.

        print('*** 1. zadatak ***')

        test_eps = [0.7, 0.4, 0.1, 0.01]

        for eps in test_eps:
            q, q_evol, old_bandit_mean = sys.run_system(
                eps=eps, ATTEMPTS_NO=ATTEMPTS_NO)
            plotter = ConvergencePlot(
                q_evol=q_evol, eps=0.1, ATTEMPTS_NO=ATTEMPTS_NO)
            plotter.plot(env=sys.env)

        # *** 2. zadatak ***
        # Naučeno Q vs epsilon = 0.

        print('*** 2. zadatak ***')

        test_eps = [0.1, 0.0]

        for eps in test_eps:
            q, q_evol, old_bandit_mean = sys.run_system(
                eps=eps, ATTEMPTS_NO=ATTEMPTS_NO)

        # *** 3. zadatak ***
        # Šta ako su karakteristike bandita promjenljive u vremenu?
        # Definišimo zakon promjene srednjih vrijednosti (može biti stohastičke ili determinističke prirode).
        # U tom slučaju ima smisla davati veću težinu trenutnim nagradama
        # kako bi se pokušala pronaći trenutno optimalna akcija.
        # Stoga se zadaje težinski faktor ALPHA,
        # koji je već implementiran i u nestacionarnom slučaju.
        # Takođe je potrebno izmijeniti implementaciju klase BanditEnvironment
        # kako bismo dodali mogućnost mijenjanja okoline.

        print('*** 3. zadatak ***')

        bandits = [Bandit(10 * (random.random() - 0.5), 5 *
                          random.random()) for _ in range(BANDITS_NO)]
        sys = System(bandits, stationary=False)
        CHANGE_AT = [4000, 6000, 9000]
        q, q_evol, old_bandit_mean = sys.run_system(
            eps=0.1, CHANGE_AT=CHANGE_AT)

        # *** 4. zadatak ***
        # Konvergencija Q vrijednosti ka srednjoj vrijednosti bandita. U ovom slučaju
        # uzimamo prethodni sistem koji je stohastičke prirode, te ćemo dobiti
        # ponašanje da Q vrijednosti pokušavaju konvergirati ka srednjoj vrijednosti.

        print('*** 4. zadatak ***')
        plotter = ConvergencePlot(
            q_evol=q_evol, eps=0.1, ATTEMPTS_NO=ATTEMPTS_NO)
        plotter.plot(env=sys.env, CHANGE_AT=CHANGE_AT,
                     old_bandit_mean=old_bandit_mean)


def main() -> None:
    unittest.main()


if __name__ == '__main__':
    print("Hi! I am testing these bandits!")
