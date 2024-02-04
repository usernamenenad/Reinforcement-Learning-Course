struct TDAlgorithm
    γ::Float64
    α::Float64
    iterations::Int
    q::Q
    results::Dict{Int,Bool}

    TDAlgorithm(; γ::Float64=1.0, α::Float64=0.1, iterations::Int=10000) = new(γ, α, iterations, Q(), Dict{Int,Bool}())
end

function SARSA(tdalg::TDAlgorithm, model::Model, actions::Vector{Action}; T::Float64=0.1)

    # Discretised states
    s::Union{State,Nothing} = nothing
    new_s::Union{State,Nothing} = nothing

    # Discretised actions
    a::Union{Action,Nothing} = nothing
    new_a::Union{Action,Nothing} = nothing

    println("Starting SARSA algorithm...")

    for i in ProgressBar(1:tdalg.iterations)
        if i - 1 % 100 == 0
            initialize_state!(model.state)
        end
        s = isnothing(s) ? discretize_state(model.state) : s
        a = isnothing(a) ? EpsGreedyPolicy(tdalg.q, s, actions) : a
        update_state!(model, a, T)

        if -x_threshold < model.state.x < x_threshold && -θ_threshold < model.state.θ < θ_threshold
            new_s = discretize_state(model.state)
            new_a = EpsGreedyPolicy(tdalg.q, new_s, actions)
            q_plus = tdalg.q[new_s, new_a]
            r = 10
            tdalg.results[i] = true
        else
            initialize_state!(model.state)
            new_a = nothing
            new_s = nothing
            q_plus = 0.0
            r = -10
            tdalg.results[i] = false
        end

        # SARSA algorithm update
        tdalg.q[s, a] = (1 - tdalg.α) * tdalg.q[s, a] + tdalg.α * (r + tdalg.γ * q_plus)

        s = new_s
        a = new_a
    end

    println("Finished SARSA algorithm!")
end
