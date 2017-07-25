mutable struct Adadelta <: Optimizer
    E_g²_t::Any
    E_Δx²_t_1::Any
    Δx²_t_1::Any
    ρ::Float64
    ϵ::Float64
end

"Construct Adadelta optimizer"
function Adadelta(shape; ρ::Float64=0.9, ϵ::Float64=1e-8)
    ρ <= 0.0 && error("ρ must be greater than 0")
    ϵ <= 0.0 && error("ϵ must be greater than 0")

    Adadelta(zeros(shape), zeros(shape), zeros(shape), ρ, ϵ)
end

function update(opt::Adadelta, g_t::Any)
    # accumulate gradient
    opt.E_g²_t = opt.ρ * opt.E_g²_t + (1 - opt.ρ) * (g_t .^ 2)

    # compute update
    RMS_g_t = sqrt.(opt.E_g²_t + opt.ϵ)
    RMS_Δx_t_1 = sqrt.(opt.E_Δx²_t_1 + opt.ϵ)
    Δx_t = RMS_Δx_t_1 .* g_t ./ RMS_g_t

    # accumulate updates
    opt.E_Δx²_t_1 = opt.ρ * opt.E_Δx²_t_1 + (1 - opt.ρ) * (Δx_t .^ 2)

    return Δx_t
end
