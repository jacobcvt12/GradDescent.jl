module GradDescent

export 
    Adagrad,
    Adadelta,
    Adam,
    update

include("AbstractOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdaDeltaOptimizer.jl")
include("AdamOptimizer.jl")

end # module
