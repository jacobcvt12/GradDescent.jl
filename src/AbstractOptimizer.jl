abstract type Optimizer
end

"Calculate change in parameters for gradient descent"
function update(opt::Optimizer, g_t::Any)
    return g_t
end
