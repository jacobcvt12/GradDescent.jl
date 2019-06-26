using Distributions, ForwardDiff
using Random: seed!
seed!(1)
Y = rand(Normal(4.0, 1.0), 100)
epochs = 200
cost(Y, μ) = loglikelihood(Normal(μ, 1.0), Y)

# adagrad
θ_adagrad = rand(Normal(), 1)
opt = Adagrad(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_adagrad)

    δ = update(opt, g)
    θ_adagrad .+= δ
end

# rmsprop
θ_rmsprop = rand(Normal(), 1)
opt = RMSprop(η=0.1)

for i in 1:(epochs*2)
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_rmsprop)

    δ = update(opt, g)
    θ_rmsprop .+= δ
end

# adam
θ_adam = rand(Normal(), 1)
opt = Adam(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_adam)

    δ = update(opt, g)
    θ_adam .+= δ
end

# adamax
θ_adamax = rand(Normal(), 1)
opt = Adamax(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_adamax)

    δ = update(opt, g)
    θ_adamax .+= δ
end

## nadam
θ_nadam = rand(Normal(), 1)
opt = Nadam(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_nadam)

    δ = update(opt, g)
    θ_nadam .+= δ
end

## vanilla
θ_vanilla = rand(Normal(), 1)
opt = VanillaGradDescent(η=0.001)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_vanilla)

    δ = update(opt, g)
    θ_vanilla .+= δ
end

## inversedecay
θ_invdec = rand(Normal(), 1)
opt = Inversedecay(κ=0.9)

for i in 1:epochs
    g = ForwardDiff.gradient(μ -> cost(Y, μ[1]), θ_invdec)

    δ = update(opt, g)
    θ_invdec .+= δ
end

@testset "Normal MLE" begin
    @test mean(Y) ≈ θ_adagrad[1] atol=1e-3
    @test mean(Y) ≈ θ_rmsprop[1] atol=1e-1
    @test mean(Y) ≈ θ_adam[1] atol=1e-3
    @test mean(Y) ≈ θ_adamax[1] atol=1e-3
    @test mean(Y) ≈ θ_nadam[1] atol=1e-3
    @test mean(Y) ≈ θ_vanilla[1] atol=1e-3
    @test mean(Y) ≈ θ_invdec[1] atol=1e-3
end
