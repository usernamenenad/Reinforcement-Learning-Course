using Distributions

const g = 9.81
const x_threshold = 5.0
const θ_threshold = deg2rad(20)

const round_precision = 3

mutable struct State
    x::Float64
    x_dot::Float64
    θ::Float64
    θ_dot::Float64
end

function initialize_state!(state::State)
    state.x = rand(Uniform(-x_threshold, x_threshold))
    state.x_dot = 0.0
    state.θ = rand(Uniform(-θ_threshold, θ_threshold))
    state.θ_dot = 0.0
end

function discretize_state(state::State)::State
    x = round(state.x; digits=round_precision)
    x_dot = round(state.x_dot; digits=round_precision)
    θ = round(state.θ; digits=round_precision)
    θ_dot = round(state.θ_dot; digits=round_precision)
    return State(x, x_dot, θ, θ_dot)
end

const Action = Float64

mutable struct Q
    q::Dict{Tuple{State,Action},Float64}

    Q(q=Dict{Tuple{State,Action},Float64}()) = new(q)
end

@inline function Base.getindex(q::Q, state::State, action::Action)::Float64
    key = (state, action)
    if !haskey(q.q, key)
        q.q[key] = rand()
    end
    return q.q[key]
end

@inline function Base.getindex(q::Q, key::Tuple{State,Action})::Float64
    if !haskey(q.q, key)
        q.q[key] = rand()
    end
    return q.q[key]
end

@inline function Base.setindex!(q::Q, value::Float64, state::State, action::Action)
    key = (state, action)
    q.q[key] = value
end

@inline function Base.setindex!(q::Q, value::Float64, key::Tuple{State,Action})
    q.q[key] = value
end
