"""
**Momentum constructor**

```julia
    Momentum(; η::Real=0.01, γ::Real=0.9)
```

Algorithm :

```math
\\begin{align*}
v_t =& \\gamma v_{t-1} + \\eta g_t\\\\
\\Delta x_t =& v_t
\\end{align*}
```
"""
mutable struct Momentum <: Optimizer
    opt_type::String
    t::Int64
    η::Float64
    γ::Float64
    v_t::AbstractArray
end

## Note the algorithm seems flawed, \η should be 1-\β and a supplementary global learning rate would be nice

function Momentum(; η::Real=0.01, γ::Real=0.9)
    @assert η > 0.0 "η must be greater than 0"
    @assert γ > 0.0 "γ must be greater than 0"

    Momentum("Momentum", 0, η, γ, [])
end

params(opt::Momentum) = "η=$(opt.η), γ=$(opt.γ)"

function update(opt::Momentum, g_t::AbstractArray{T}) where {T<:Real}
    # resize squares of gradients
    if opt.t == 0
        opt.v_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    opt.v_t = opt.γ * opt.v_t + opt.η * g_t

    return opt.v_t
end
