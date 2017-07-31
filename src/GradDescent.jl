"""
*Gradient Descent optimizers for Julia.*

# Introduction

This package abstracts the "boilerplate" code necessary for gradient descent. Gradient descent is "a way to minimize an objective function ``J(θ)`` parameterized by a model's parameters ``θ ∈ Rᵈ``" (Ruder 2017). Gradient descent finds ``θ`` which minizes ``J`` by iterating over the following update

``θ = θ - η ∇J(θ)``

until convergence of ``θ``. Certainly, the gradient calculation is model specific, however the learning rate ``η`` (at a given iteration) is not. Instead there are many different gradient descent variants which determine the learning rate. Each type of gradient descent optimizer has its own pros/cons. For most of these optimizers, the calculation of the learning rate is based on the value of the gradient (evaluated at a particular ``θ``) and a few (unrelated to the model) hyperparameters. 

The purpose of this package is to allow the user to focus on the calculation of the gradients and not worry about the code for the gradient descent optimizer. I envision a user implementing his/her gradients, experimenting with various optimizers, and modifying the gradients as necessary.

# Examples

Here I demonstrate a very simple example - minimizing ``x²``. In this example, I use "Adagrad", a common gradient descent optimizer.

```julia
using GradDescent

# objective function and gradient of objective function
J(x) = x ^ 2
dJ(x) = 2 * x

# number of epochs
epochs = 1000

# instantiation of Adagrad optimizer with learning rate of 1.0
# note that this learning rate is likely to high for a
# high dimensional case
opt = Adagrad(η=1.0)

# initial value for x (usually initialized with a random value)
x = 20.0

for i in 1:epochs
    # calculate the gradient wrt to the current x
    g = dJ(x)

    # change to the current x
    δ = update(opt, g)
    x -= δ
end
```

Next I demonstrate a more common example - determining the coefficients of a linear model. Here I use "Adam" an extension of "Adagrad". In this example, we minimize the mean squared error of the predicted outcome and the actual outcome. The parameter space is the coefficients of the regression model.

```julia
using GradDescent, Distributions, ReverseDiff

srand(1) # set seed
n = 1000 # number of observations
d = 10   # number of covariates
X = rand(Normal(), n, d) # simulated covariates
b = rand(Normal(), d)    # generated coefficients
ϵ = rand(Normal(0.0, 0.1), n) # noise
Y = X * b + ϵ # observed outcome
obj(Y, X, b) = mean((Y - X * b) .^ 2) # objective to minimize

epochs = 100 # number of epochs

θ = rand(Normal(), d) # initialize model parameters
opt = Adam(α=1.0)  # initalize optimizer with learning rate 1.0

for i in 1:epochs
    # here we use automatic differentiation to calculate 
    # the gradient at a value
    # an analytically derived gradient is not required
    g = ReverseDiff.gradient(θ -> obj(Y, X, θ), θ)

    δ = update(opt, g)
    θ -= δ
end
```

"""
module GradDescent

export 
    Momentum,
    Adagrad,
    Adadelta,
    RMSprop,
    Adam,
    update,
    t

include("AbstractOptimizer.jl")
include("MomentumOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdaDeltaOptimizer.jl")
include("RMSpropOptimizer.jl")
include("AdamOptimizer.jl")

end # module
