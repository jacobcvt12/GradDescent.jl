opt = Adagrad()

A = [1.0 2.0; 3.0 4.0]
B = [5. 6. 7.; 8. 9. 0.]
g_t = [A, B]

@testset "Optimizer List" begin
    @test_nowarn update(opt, g_t)
end
