module GradDescent

using LinearAlgebra
export
    Optimizer,
    VanillaGradDescent,
    Inversedecay,
    Momentum,
    Adagrad,
    Adadelta,
    RMSprop,
    Adam,
    Adamax,
    Nadam,
    update,
    t

include("AbstractOptimizer.jl")
include("VanillaGradDescent.jl")
include("InverseDecayOptimizer.jl")
include("MomentumOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdaDeltaOptimizer.jl")
include("RMSpropOptimizer.jl")
include("AdamOptimizer.jl")
include("AdamaxOptimizer.jl")
include("NadamOptimizer.jl")

end # module
