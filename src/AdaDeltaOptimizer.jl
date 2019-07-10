"""
**Adadelta constructor**

```julia
    Adadelta(; ρ::Float64=0.9, ϵ::Float64=1e-8)
```

Algorithm :

```math
\\begin{align*}
E[g^2]_t =& \\rho  E[g^2]_{t-1}+(1-\\rho)g^2_t\\\\
\\text{RMS}[g]_t =& \\sqrt{E[g^2]_t + \\epsilon}\\\\
\\text{RMS}[\\Delta x]_{t-1} =& \\sqrt{E[\\Delta x^2]_{t-1} + \\epsilon}\\\\
\\Delta x_t =& \\text{RMS}[Δx]_{t-1} * g_t / \\text{RMS}[g]_t\\\\
E[\\Delta x^2]_{t} =& \\rho E[\\Delta x^2]_{t} + (1 - \\rho) \\Delta x^2_t\\\\
\\end{align*}
```

[Algorithm Reference](https://arxiv.org/abs/1212.5701)
"""
mutable struct Adadelta <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    ρ::Float64
    E_g²_t::AbstractArray
    E_Δx²_t_1::AbstractArray
    Δx²_t_1::AbstractArray
end

function Adadelta(; ρ::Real=0.9, ϵ::Real=1e-8)
    @assert ρ > 0.0 "ρ must be greater than 0"
    @assert ϵ > 0.0 "ϵ must be greater than 0"

    Adadelta("Adadelta", 0, ϵ, ρ, [], [], [])
end

params(opt::Adadelta) = "ϵ=$(opt.ϵ), ρ=$(opt.ρ)"

function update(opt::Adadelta, g_t::AbstractArray{T}) where {T<:Real}
    # resize accumulated and squared updates
    if opt.t == 0
        opt.E_g²_t = zero(g_t)
        opt.E_Δx²_t_1  = zero(g_t)
        opt.Δx²_t_1 = zero(g_t)
    end

    # accumulate gradient
    opt.E_g²_t = opt.ρ * opt.E_g²_t + (one(T) - opt.ρ) * (g_t .^ 2)

    # compute update
    RMS_g_t = sqrt.(opt.E_g²_t .+ opt.ϵ)
    RMS_Δx_t_1 = sqrt.(opt.E_Δx²_t_1 .+ opt.ϵ)
    Δx_t = RMS_Δx_t_1 .* g_t ./ RMS_g_t

    # accumulate updates
    opt.E_Δx²_t_1 = opt.ρ * opt.E_Δx²_t_1 + (one(T) - opt.ρ) * (Δx_t .^ 2)

    return Δx_t
end

"""

"""
