using Distributions, ForwardDiff

srand(1)
n = 1000
d = 10
X = rand(Normal(), n, d)
b = rand(Normal(), d)
ϵ = rand(Normal(0.0, 0.1), n)
Y = X * b + ϵ
obj(Y, X, b) = mean((Y - X * b) .^ 2)

epochs = 100

# momentum
θ_momentum = rand(Normal(), d)
opt = Momentum(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_momentum)

    δ = update(opt, g)
    θ_momentum -= δ
end

# adagrad
θ_adagrad = rand(Normal(), d)
opt = Adagrad(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adagrad)

    δ = update(opt, g)
    θ_adagrad -= δ
end

# rmsprop
θ_rmsprop = rand(Normal(), d)
opt = RMSprop(η=0.1)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_rmsprop)

    δ = update(opt, g)
    θ_rmsprop -= δ
end

# adam
θ_adam = rand(Normal(), d)
opt = Adam(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adam)

    δ = update(opt, g)
    θ_adam -= δ
end

# adamax
θ_adamax = rand(Normal(), d)
opt = Adamax(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adamax)

    δ = update(opt, g)
    θ_adamax -= δ
end

@testset "Linear Regression" begin
    @test b ≈ θ_momentum atol=1e-1
    @test b ≈ θ_adagrad atol=1e-1
    @test b ≈ θ_adam atol=1e-1
    @test b ≈ θ_adamax atol=1e-1
end
