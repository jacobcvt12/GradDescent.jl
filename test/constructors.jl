@testset "Constructors" begin
    @test_nowarn opt = Adagrad()
    @test_nowarn opt = Adadelta()
    @test_nowarn opt = Adam()
end
