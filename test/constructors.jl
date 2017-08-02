@testset "Constructors" begin
    @test_nowarn opt = Momentum()
    @test_nowarn opt = Adagrad()
    @test_nowarn opt = Adadelta()
    @test_nowarn opt = RMSprop()
    @test_nowarn opt = Adam()
    @test_nowarn opt = Adamax()
end
