mutable struct Inversedecay <: Optimizer
    opt_type::String
    t::Int64
    t₀::Float64
    η::Float64
end

"""
    Construct an Inverse Decay Gradient Descent optimizer
"""
function Inversedecay(; t0::Real=1, κ::Real=0.51)
    @assert t0 <= 0.0 "t0 must be greater than 0"
    @assert (κ <= 0.5 || κ > 1.0) "κ argument is in (0.5,1]"

    InverseDecay("Inversedecay", 0, t0, κ)
end

params(opt::Inversedecay) = "t₀=$(opt.t₀), κ=$(opt.κ)"

function update(opt::Inversedecay, g_t::AbstractArray{T}) where {T<:Real}
    # update timestep
    opt.t += 1
    return (opt.t₀+opt.t)^(-opt.κ) * g_t
end
