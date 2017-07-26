abstract type Optimizer
end

"Calculate change in parameters for gradient descent"
update(opt::Optimizer, g_t::Array{Float64}) = error("not implemented")
update(opt::Optimizer, g_t::Float64) = update(opt::Optimizer, [g_t])

"Number of epochs run"
t(opt::Optimizer) = opt.t

"Type of gradient descent optimizer"
optimizer(opt::Optimizer) = opt.opt_type

"Print summary"
function print(opt::Optimizer) 
    print("Optimizer: $(optimizer(opt))")
end
