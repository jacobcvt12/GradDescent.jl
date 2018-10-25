abstract type Optimizer
end

"Calculate change in parameters for gradient descent"
update(opt::Optimizer, g_t::AbstractArray{T,N}) where {T<:Real,N} = error("not implemented")
update(opt::Optimizer, g_t::Real) = update(opt::Optimizer, [g_t])[1]

"Number of epochs run"
t(opt::Optimizer) = opt.t

optimizer(opt::Optimizer) = opt.opt_type

params(opt::Optimizer) = error("not implemented")

"Print summary"
function Base.show(io::IO, opt::Optimizer)
    print("$(optimizer(opt))(t=$(t(opt::Optimizer)), $(params(opt)))")
end

"Deep copy an optimizer"
function Base.deepcopy(opt::Optimizer)
    f = length(fieldnames(opt))
    copied_params = [deepcopy(getfield(opt, k)) for k = 1:f]
    typeof(opt)(copied_params...)
end
