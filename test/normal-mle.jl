using Distributions, ForwardDiff

srand(1)
Y = rand(Normal(4.0, 1.0), 100)
epochs = 100
cost(Y, μ) = loglikelihood(Normal(μ, 1.0), Y)

# adagrad
θ = rand(Normal(), 1)
opt = Adagrad(1, η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ)

    δ = update(opt, g)
    θ += δ
end

@test mean(Y) ≈ θ[1] atol=1e-5
