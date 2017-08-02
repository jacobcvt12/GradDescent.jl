cost(x) = x ^ 2
dcost(x) = 2 * x

srand(1)
epochs = 2000

# momentum
x_momentum = rand(1)
opt = Momentum()

for i in 1:epochs
    g = dcost(x_momentum)

    δ = update(opt, g)
    x_momentum -= δ
end

# adagrad
x_adagrad = rand(1)
opt = Adagrad()

for i in 1:epochs
    g = dcost(x_adagrad)

    δ = update(opt, g)
    x_adagrad -= δ
end

# adadelta
x_adadelta = rand(1)
opt = Adadelta()

for i in 1:epochs
    g = dcost(x_adadelta)

    δ = update(opt, g)
    x_adadelta -= δ
end

# rmsprop
x_rmsprop = rand(1)
opt = RMSprop()

for i in 1:epochs
    g = dcost(x_rmsprop)

    δ = update(opt, g)
    x_rmsprop -= δ
end

# adam
x_adam = rand(1)
opt = Adam()

for i in 1:epochs
    g = dcost(x_adam)

    δ = update(opt, g)
    x_adam -= δ
end

# adamax
x_adamax = rand(1)
opt = Adamax()

for i in 1:epochs
    g = dcost(x_adamax)

    δ = update(opt, g)
    x_adamax -= δ
end

# nadam
x_nadam = rand(1)
opt = Nadam()

for i in 1:epochs
    g = dcost(x_nadam)

    δ = update(opt, g)
    x_nadam -= δ
end

@testset "Quadatric" begin
    @test 0.0 ≈ x_momentum[1] atol=5e-2
    @test 0.0 ≈ x_adagrad[1] atol=5e-2
    @test 0.0 ≈ x_adadelta[1] atol=5e-2
    @test 0.0 ≈ x_rmsprop[1] atol=5e-2
    @test 0.0 ≈ x_adam[1] atol=5e-2
    @test 0.0 ≈ x_adamax[1] atol=5e-2
    @test 0.0 ≈ x_nadam[1] atol=5e-2
end
