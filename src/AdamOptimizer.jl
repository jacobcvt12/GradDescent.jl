mutable struct Adam <: Optimizer
    α::Float64
    β₁::Float64
    β₂::Float64
    ϵ::Float64
    t::Int64
    m_t::Matrix{Float64}
    v_t::Matrix{Float64}
end

"Construct Adam optimizer"
function Adam(α=0.001, β₁=0.9, β₂=0.999, ϵ=10e-8)
    m_t = zeros(1)'
    v_t = zeros(1)'

    Adam(α, β₁, β₂, ϵ, 0, m_t, v_t)
end

function update(opt::Adam, g_t::Any)
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zeros(length(g_t))'
        opt.v_t = zeros(length(g_t))'
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (1. - opt.β₁) * g_t

    # update biased second raw moment estimate
    opt.v_t = opt.β₂ * opt.v_t + (1. - opt.β₂) * ((g_t) .^2)

    # compute bias corrected first moment estimate
    m̂_t = opt.m_t / (1. - opt.β₁^opt.t)

    # compute bias corrected second raw moment estimate
    v̂_t = opt.v_t / (1. - opt.β₂^opt.t)

    # apply update
    ρ = opt.α * m̂_t ./ (sqrt.(v̂_t + opt.ϵ))

    return ρ
end
