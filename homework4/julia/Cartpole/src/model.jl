mutable struct Model
    m::Float64
    M::Float64
    l::Float64
    state::State

    Model(m::Float64, M::Float64, L::Float64, state::State=State(0.0, 0.0, 0.0, 0.0)) = new(m, M, L / 2, state)
end

function F(model::Model, u::Action)::Float64
    m = model.m
    M = model.M
    l = model.l

    θ = model.state.θ
    θ_dot = model.state.θ_dot

    num = 4 * u - m * sin(θ) * (3 * g * cos(θ) - 4 * l * θ_dot^2)
    den = 4 * (m + M) - 3 * m * cos(θ)^2

    return num / den
end

function G(model::Model, u::Action)::Float64
    m = model.m
    M = model.M
    l = model.l

    θ = model.state.θ
    θ_dot = model.state.θ_dot

    num = (m + M) - g * sin(θ) - cos(θ) * (u + m * l * sin(θ) * θ_dot^2)
    den = l * (4 / 3 * (m + M) - m * cos(θ)^2)

    return num / den
end

function update_state!(model::Model, u::Action, T::Float64)
    model.state.x += T * model.state.x_dot
    model.state.x_dot += T * F(model, u)
    model.state.θ += T * model.state.θ_dot
    model.state.θ_dot += T * G(model, u)
end