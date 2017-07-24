module GradDescent

export 
    Adagrad,
    Adam,
    update

include("AbstractOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdamOptimizer.jl")

end # module
