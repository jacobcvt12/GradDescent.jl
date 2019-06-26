mutable struct Nadam <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    η::Float64
    β₁::Float64
    β₂::Float64
    m_t::AbstractArray
    v_t::AbstractArray
end

"Construct Nadam optimizer"
function Nadam(;η::Real=0.001, β₁::Real=0.9, β₂::Real=0.999, ϵ::Real=10e-8)
    @assert η > 0.0 "η must be greater than 0"
    @assert β₁ > 0.0 "β₁ must be greater than 0"
    @assert β₂ > 0.0 "β₂ must be greater than 0"
    @assert ϵ > 0.0 "ϵ must be greater than 0"

    Nadam("Nadam", 0, ϵ, η, β₁, β₂, [], [])
end

params(opt::Nadam) = "ϵ=$(opt.ϵ), η=$(opt.η), β₁=$(opt.β₁), β₂=$(opt.β₂)"

function update(opt::Nadam, g_t::AbstractArray{T}) where {T<:Real}
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zero(g_t)
        opt.v_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (one(T) - opt.β₁) * g_t

    # update biased second raw moment estimate
    opt.v_t = opt.β₂ * opt.v_t + (one(T) - opt.β₂) * ((g_t) .^2)

    # compute bias corrected first moment estimate
    m̂_t = opt.m_t / (one(T) - opt.β₁^opt.t)

    # compute bias corrected second raw moment estimate
    v̂_t = opt.v_t / (one(T) - opt.β₂^opt.t)

    # apply update
    ρ = opt.η ./ (sqrt.(v̂_t .+ opt.ϵ))
    ρ .*= (opt.β₁ * m̂_t + (one(T) - opt.β₁) * g_t / (one(T) - opt.β₁^opt.t))

    return ρ
end
