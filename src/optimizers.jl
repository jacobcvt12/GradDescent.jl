abstract type Optimizer
end

struct Adagrad <: Optimizer
    grad::Any
    η::Float64
end
