mutable struct VanillaGradDescent <: Optimizer
    opt_type::String
    t::Int64
    η::Float64
end

"Construct Vanilla Gradient Descent optimizer"
function VanillaGradDescent(; η::Real=0.1)
    @assert η > 0.0 "η must be greater than 0"

    VanillaGradDescent("Vanilla Gradient Descent", 0, η)
end

params(opt::VanillaGradDescent) = "η=$(opt.η)"

function update(opt::VanillaGradDescent, g_t::AbstractArray{T}) where {T<:Real}
    # update timestep
    opt.t += 1
    return opt.η * g_t
end
