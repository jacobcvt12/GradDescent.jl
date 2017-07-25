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

# adagrad
θ_adagrad = rand(Normal(), d)
opt = Adagrad(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adagrad)

    δ = update(opt, g)
    θ_adagrad -= δ
end

# adam
θ_adam = rand(Normal(), d)
opt = Adam(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adam)

    δ = update(opt, g)
    θ_adam -= δ
end

@testset "Linear Regression" begin
    @test b ≈ θ_adagrad atol=1e-1
    @test b ≈ θ_adam atol=1e-1
end
