mutable struct Adamax <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    α::Float64
    β₁::Float64
    β₂::Float64
    m_t::AbstractArray
    u_t::AbstractArray
end

"""
    Adamax Optimizer
    `Adamax(;α=0.002, β₁=0.9, β₂=0.999, ϵ=10e-8)`
    Algorithm
    ```
        m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t
        v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2
        \\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}
        \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}
        \theta_{t+1} = \theta_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t}+\\epsilon}\\hat{m}_t
    ```
    [Reference](https://arxiv.org/abs/1412.6980)
"""
function Adamax(;α=0.002, β₁=0.9, β₂=0.999, ϵ=10e-8)
    m_t = zeros(Float64,1)'
    u_t = zeros(Float64,1)'

    Adamax("Adamax", 0, ϵ, α, β₁, β₂, m_t, u_t)
end

params(opt::Adamax) = "ϵ=$(opt.ϵ), α=$(opt.α), β₁=$(opt.β₁), β₂=$(opt.β₂)"

function update(opt::Adamax, g_t::AbstractArray{T}) where {T<:Real}
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zero(g_t)
        opt.u_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (1. - opt.β₁) * g_t

    # update the exponentially weighted infinity norm
    opt.u_t = max.(opt.β₂ * opt.u_t, abs.(g_t))

    # update parameters
    ρ = (opt.α / (1- opt.β₁^opt.t)) * opt.m_t ./ opt.u_t

    return ρ
end
