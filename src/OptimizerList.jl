mutable struct OptimizerList <: Optimizer
    opts::Array{Optimzer}
end

"Promote Optimizer to OptimizerList"
function update(opt::Optimizer, 
                g_t::Array{Array{Float64}})
    # length of list
    n = length(g_t)

    opt = OptimizerList(repeat(opt, n))
end
