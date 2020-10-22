function _test_imput(A, ob::Vector, b::Vector)
    IM = ImputingMatrix(A, ob)
    @test IM*b == A*ob
end

function _test_imput(A, ob::Vector, t=10, p=0.2)
    for _ in 1:t
        b = [rand() < 0.2 ? missing : x for x in ob]
        idcs = rand([true, false], length(ob))
        _test_imput(A, ob, b)
    end
end

@testset "correct imputing behavior" begin
    _test_imput(rand(3,3), rand(3))
    _test_imput(reshape(1:9 |> collect, (3,3)), rand(1:3, 3))
    _test_imput(rand(3,3), rand(3), fill(missing, 3))
    _test_imput(reshape(1:9 |> collect, (3,3)), rand(1:3, 3), fill(missing, 3))

    A = [1 2; 3 4]
    ψ = [2, 1]
    IM = ImputingMatrix(A, ψ)
    B1 = [4 3; 2 1]
    B2 = [missing missing; missing missing]
    B3 = [missing 3; 1 missing]
    B4 = [3 missing; missing 1]

    @test IM*B1 == A*B1
    @test IM*B2 == [A*ψ A*ψ]
    @test IM*B3 == A*[2 3; 1 1]
    @test IM*B4 == A*[3 2; 1 1]
end
