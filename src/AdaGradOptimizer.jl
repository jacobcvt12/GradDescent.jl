mutable struct Adagrad <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    η::Float64
    G_t::AbstractArray
end

"Construct Adagrad optimizer"
function Adagrad(; η::Real=0.01, ϵ::Real=1e-8)
    @assert η > 0.0 "η must be greater than 0"
    @assert ϵ > 0.0 "ϵ must be greater than 0"

    Adagrad("Adagrad", 0, ϵ, η, [])
end

params(opt::Adagrad) = "ϵ=$(opt.ϵ), η=$(opt.η)"

function update(opt::Adagrad, g_t::AbstractArray{T}) where {T<:Real}
    # resize squares of gradients
    if opt.t == 0
        opt.G_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # accumulate squares of gradients
    opt.G_t .+= (g_t .^ 2)

    δ = opt.η ./ (sqrt.(opt.G_t .+ opt.ϵ)) .* g_t

    return δ
end
