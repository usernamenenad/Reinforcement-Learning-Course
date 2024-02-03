module Cartpole

export Model

g = 9.81

mutable struct State
    x::Float64
    x_dot::Float64
    θ::Float64
    θ_dot::Float64
end

struct Model
    m::Float64
    M::Float64
    l::Float64

    # Constructor
    Model(m::Float64, M::Float64, L::Float64) = m < 0 || M < 0 || L < 0 ? error("A physical quantity like mass or length cannot have values less than zero!") : new(m, M, L / 2)

    # Helper function
    function F!(s::State, T::Float64, u::Float64)::Float64
        num = 4u - m * sin(s.θ) * (3g * cos(s.θ_dot) - 4l * s.θ_dot)
        den = 4(m + M) - 3m * (cos(s.θ))^2
        return num / den
    end

    # Helper function
    function G!(s::State, T::Float64, u::Float64)::Float64
        num = (m + M)g * sin(s.θ) - cos(s.θ) * (u + m * l * sin(s.θ) * s.θ_dot^2)
        den = l * (4 / 3 * (m + M) - m * (cos(s.θ))^2)
        return num / den
    end

    #A model output
    function update!(s::State, T::Float64, u::Float64)::Nothing
        s.x += T * s.x_dot
        s.x_dot += T * F!(s, T, u)
        s.θ += T * s.θ_dot
        s.θ_dot += T * G!(s, T, u)
    end


end



end