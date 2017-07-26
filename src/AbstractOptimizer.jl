abstract type Optimizer
end

"Calculate change in parameters for gradient descent"
update(opt::Optimizer, g_t::Array{Float64}) = error("not implemented")
update(opt::Optimizer, g_t::Float64) = update(opt::Optimizer, [g_t])
