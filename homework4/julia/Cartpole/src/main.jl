using Cartpole

model = Model(1.0, 1.0, 1.0)
alg = TDAlgorithm(γ=0.9)

SARSA(alg, model, [-1.0, 0.0, 1.0])

log_results(alg.results)