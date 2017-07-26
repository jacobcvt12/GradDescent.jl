module GradDescent

export 
    Momentum,
    Adagrad,
    Adadelta,
    RMSprop,
    Adam,
    update,
    t,
    optimizer,
    print

include("AbstractOptimizer.jl")
include("MomentumOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdaDeltaOptimizer.jl")
include("RMSpropOptimizer.jl")
include("AdamOptimizer.jl")

end # module
