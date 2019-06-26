using Distributions, ForwardDiff
using Random: seed!

seed!(1)
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
    global θ_momentum -= δ
end

# adagrad
θ_adagrad = rand(Normal(), d)
opt = Adagrad(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adagrad)

    δ = update(opt, g)
    global θ_adagrad -= δ
end

# rmsprop
θ_rmsprop = rand(Normal(), d)
opt = RMSprop(η=0.1)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_rmsprop)

    δ = update(opt, g)
    global θ_rmsprop -= δ
end

# adam
θ_adam = rand(Normal(), d)
opt = Adam(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adam)

    δ = update(opt, g)
    global θ_adam -= δ
end

# adamax
θ_adamax = rand(Normal(), d)
opt = Adamax(α=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_adamax)

    δ = update(opt, g)
    global θ_adamax -= δ
end

# nadam
θ_nadam = rand(Normal(), d)
opt = Nadam(η=1.0)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_nadam)

    δ = update(opt, g)
    global θ_nadam -= δ
end

## vanilla
θ_vanilla = rand(Normal(), d)
opt = VanillaGradDescent(η=0.001)

for i in 1:20*epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_vanilla)

    δ = update(opt, g)
    global θ_vanilla -= δ
end


## inversedecay
θ_invdec = rand(Normal(), d)
opt = Inversedecay(κ=0.9)

for i in 1:epochs
    g = ForwardDiff.gradient(θ -> obj(Y, X, θ), θ_invdec)

    δ = update(opt, g)
    global θ_invdec -= δ
end

##

@testset "Linear Regression" begin
    @test b ≈ θ_momentum atol=1e-1
    @test b ≈ θ_adagrad atol=1e-1
    @test b ≈ θ_adam atol=1e-1
    @test b ≈ θ_adamax atol=1e-1
    @test b ≈ θ_nadam atol=1e-1
    @test b ≈ θ_vanilla atol=1e-1
    @test b ≈ θ_invdec atol=1e-1
end
