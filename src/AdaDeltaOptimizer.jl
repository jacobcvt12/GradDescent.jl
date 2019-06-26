mutable struct Adadelta <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    ρ::Float64
    E_g²_t::AbstractArray
    E_Δx²_t_1::AbstractArray
    Δx²_t_1::AbstractArray
end

"Construct Adadelta optimizer"
function Adadelta(; ρ::Float64=0.9, ϵ::Float64=1e-8)
    @assert ρ <= 0.0 "ρ must be greater than 0"
    @assert ϵ <= 0.0 "ϵ must be greater than 0"

    Adadelta("Adadelta", 0, ϵ, ρ, zeros(1), zeros(1), zeros(1))
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
    opt.E_g²_t = opt.ρ * opt.E_g²_t + (1 - opt.ρ) * (g_t .^ 2)

    # compute update
    RMS_g_t = sqrt.(opt.E_g²_t .+ opt.ϵ)
    RMS_Δx_t_1 = sqrt.(opt.E_Δx²_t_1 .+ opt.ϵ)
    Δx_t = RMS_Δx_t_1 .* g_t ./ RMS_g_t

    # accumulate updates
    opt.E_Δx²_t_1 = opt.ρ * opt.E_Δx²_t_1 + (1 - opt.ρ) * (Δx_t .^ 2)

    return Δx_t
end
