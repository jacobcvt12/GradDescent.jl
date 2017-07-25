module GradDescent

export 
    Momentum,
    Adagrad,
    Adadelta,
    Adam,
    update

include("AbstractOptimizer.jl")
include("MomentumOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdaDeltaOptimizer.jl")
include("AdamOptimizer.jl")

end # module
