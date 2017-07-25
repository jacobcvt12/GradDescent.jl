using Distributions, ForwardDiff

srand(1)
Y = rand(Normal(4.0, 1.0), 100)
epochs = 200
cost(Y, μ) = loglikelihood(Normal(μ, 1.0), Y)

# adagrad
θ_adagrad = rand(Normal(), 1)
opt = Adagrad(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_adagrad)

    δ = update(opt, g)
    θ_adagrad += δ
end

# adadelta
θ_adadelta = rand(Normal(), 1)
opt = Adadelta()

for i in 1:(epochs*20) # no "questioningly" increase learning rate for adadelta
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_adadelta)

    δ = update(opt, g)
    θ_adadelta += δ
end

θ_adadelta

# adam
θ_adam = rand(Normal(), 1)
opt = Adam(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_adam)

    δ = update(opt, g)
    θ_adam += δ
end

@testset "Normal MLE" begin
    @test mean(Y) ≈ θ_adagrad[1] atol=1e-3
    @test mean(Y) ≈ θ_adadelta[1] atol=1e-0
    @test mean(Y) ≈ θ_adam[1] atol=1e-3
end
