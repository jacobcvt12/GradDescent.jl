mutable struct Adagrad <: Optimizer
    G_t::Any
    η::Float64
    ϵ::Float64
end

function Adagrad(shape; η::Float64=0.01, ϵ::Float64=1e-8)
    η <= 0.0 && error("η must be greater than 0")
    ϵ <= 0.0 && error("ϵ must be greater than 0")

    Adagrad(zeros(shape), η, ϵ)
end

function update(opt::Adagrad, g_t::Any)
    opt.G_t += (g_t .^ 2)

    δ = opt.η ./ (sqrt.(opt.G_t + opt.ϵ)) .* g_t

    return δ
end
