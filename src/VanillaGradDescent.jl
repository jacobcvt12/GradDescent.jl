"""
**Basic gradient descent with fixed learning rate**
```julia
VanillaGradDescent(; η::Real=0.01)
```

Algorithm
```math
\\Delta x_t = \\eta g_t
```

"""
mutable struct VanillaGradDescent <: Optimizer
    opt_type::String
    t::Int64
    η::Float64
end

function VanillaGradDescent(; η::Real=0.01)
    @assert η > 0.0 "η must be greater than 0"

    VanillaGradDescent("Vanilla Gradient Descent", 0, η)
end

params(opt::VanillaGradDescent) = "η=$(opt.η)"

function update(opt::VanillaGradDescent, g_t::AbstractArray{T}) where {T<:Real}
    # update timestep
    opt.t += 1
    return opt.η * g_t
end
