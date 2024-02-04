module Cartpole

using Distributions
using Plots
using ProgressBars

include("utils.jl")
include("model.jl")
include("policies.jl")
include("td.jl")
include("info.jl")

export Model
export TDAlgorithm, SARSA
export log_results

end # module Cartpole
