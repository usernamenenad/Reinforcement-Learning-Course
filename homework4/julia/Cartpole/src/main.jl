using Cartpole

model = Cartpole.Model(1.0, 1.0, 1.0)
alg = Cartpole.TDAlgorithm()

Cartpole.SARSA(alg, model, [-1.0, 0.0, 1.0])