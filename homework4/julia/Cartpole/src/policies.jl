function GreedyPolicy(q::Q, state::State, actions::Vector{Action})::Action
    to_max::Dict{Action,Float64} = Dict{Action,Float64}()
    for action ∈ actions
        to_max[action] = q[state, action]
    end
    return findmax(to_max)[2]
end

function RandomPolicy(actions::Vector{Action})::Action
    return rand(actions)
end

function EpsGreedyPolicy(q::Q, state::State, actions::Vector{Action}, ϵ::Float64=0.1)::Action
    return ϵ > rand() ? GreedyPolicy(q, state, actions) : RandomPolicy(actions)
end
