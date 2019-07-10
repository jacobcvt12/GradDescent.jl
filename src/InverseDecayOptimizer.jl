"""
**Inversedecay constructor**
```julia
    Inversedecay(; t0::Real=1, κ::Real=0.51)
```

Algorithm:
```math
\\Delta x_t = (t+t_0)^{-\\kappa}g_t
```

Simple learning rate respecting the [Robbins-Monro conditions](https://en.wikipedia.org/wiki/Stochastic_approximation#Robbins%E2%80%93Monro_algorithm) for κ ∈ (0.5,1]
"""
mutable struct Inversedecay <: Optimizer
    opt_type::String
    t::Int64
    t₀::Float64
    κ::Float64
end

function Inversedecay(; t0::Real=1, κ::Real=0.51)
    @assert t0 > 0.0 "t0 must be greater than 0"
    @assert (κ > 0.5 || κ <= 1.0) "κ argument is in (0.5,1]"

    Inversedecay("Inversedecay", 0, t0, κ)
end

params(opt::Inversedecay) = "t₀=$(opt.t₀), κ=$(opt.κ)"

function update(opt::Inversedecay, g_t::AbstractArray{T}) where {T<:Real}
    # update timestep
    opt.t += 1
    return (opt.t₀+opt.t)^(-opt.κ) * g_t
end
