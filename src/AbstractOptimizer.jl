abstract type Optimizer end

"Calculate change in parameters for gradient descent"
update(opt::Optimizer, g_t::AbstractArray{T}) where {T<:Real} = error("not implemented")
update(opt::Optimizer, g_t::Real) = update(opt::Optimizer, [g_t])[1]

"Number of epochs run"
t(opt::Optimizer) = opt.t

optimizer(opt::Optimizer) = opt.opt_type

params(opt::Optimizer) = error("not implemented")

"Print summary"
function Base.show(io::IO, opt::Optimizer)
    print(io,"$(optimizer(opt))(t=$(t(opt::Optimizer)), $(params(opt)))")
end

"Deep copy an optimizer"
function Base.deepcopy(opt::O) where {O<:Optimizer}
    O([deepcopy(getfield(opt,f)) for f in fieldnames(O)]...)
end
