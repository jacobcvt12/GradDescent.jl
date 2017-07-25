cost(x) = x ^ 2
dcost(x) = 2 * x

srand(1)
epochs = 1000

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

# adam
x_adam = rand(1)
opt = Adam()

for i in 1:epochs
    g = dcost(x_adam)

    δ = update(opt, g)
    x_adam -= δ
end

@testset "Quadatric" begin
    @test 0.0 ≈ x_adagrad[1] atol=5e-2
    @test 0.0 ≈ x_adadelta[1] atol=5e-2
    @test 0.0 ≈ x_adam[1] atol=5e-2
end
