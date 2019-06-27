mutable struct Adamax <: Optimizer
    opt_type::String
    t::Int64
    α::Float64
    β₁::Float64
    β₂::Float64
    m_t::AbstractArray
    u_t::AbstractArray
end

"""
Adamax Optimizer

```julia
Adamax(;α=0.002, β₁=0.9, β₂=0.999, ϵ=10e-8)
```
Algorithm

```math
\\begin{align*}
m_t =& \\beta_1 m_{t-1} + (1-\\beta_1)g_t\\\\
u_t =& \\max(\\beta_2 u_{t}, |g_t|)\\\\
\\Delta x_t =& \\frac{\\alpha}{(1-\\beta_1^t)}\\frac{m_t}{u_t}
\\end{align*}
```

[Algorithm Reference](https://arxiv.org/abs/1412.6980)
"""
function Adamax(;α::Real=0.002, β₁::Real=0.9, β₂::Real=0.998)
    @assert α > 0.0 "α must be greater than 0"
    @assert β₁ > 0.0 "β₁ must be greater than 0"
    @assert β₂ > 0.0 "β₂ must be greater than 0"

    Adamax("Adamax", 0, α, β₁, β₂, [], [])
end

params(opt::Adamax) = "α=$(opt.α), β₁=$(opt.β₁), β₂=$(opt.β₂)"

function update(opt::Adamax, g_t::AbstractArray{T}) where {T<:Real}
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zero(g_t)
        opt.u_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (one(T) - opt.β₁) * g_t

    # update the exponentially weighted infinity norm
    opt.u_t = max.(opt.β₂ * opt.u_t, abs.(g_t))

    # update parameters
    ρ = (opt.α / (one(T) - opt.β₁^opt.t)) * opt.m_t ./ opt.u_t

    return ρ
end
