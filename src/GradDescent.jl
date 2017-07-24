module GradDescent

export 
    Adagrad,
    update

include("AbstractOptimizer.jl")
include("AdaGradOptimizer.jl")

end # module
