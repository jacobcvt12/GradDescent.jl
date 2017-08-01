mutable struct OptimizerList <: Optimizer
    opts::Array{Optimzer}
end

"Promote Optimizer to OptimizerList"
function update(opt::Optimizer, 
                g_t::Array{Array{Float64}})
    # length of list
    n = length(g_t)

    opt = OptimizerList([deepcopy(opt) for k in 1:n])

    update(opt, g_t)
end

function update(opt::OptimizerList, 
                g_t::Array{Array{Float64}}) 
    # length of list
    n = length(g_t)

    δ = Array{Array{Float64, 2}, 1}(n)

    for i in 1:n
        δ[i] = update(opt.opts[i], g_t[i])
    end

    return δ
end
