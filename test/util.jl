md = fill("metadata", 4)
an1 = ArrayNode(rand(Float32, 3, 4))
b = BagNode(an1, [1:4, 0:-1], md)
an2 = ArrayNode(randn(Float32, 5, 4))
wb = WeightedBagNode(an2, [1:2, 3:4], rand(Float32, 4), md)
pn = ProductNode(; b, wb)
an3 = ArrayNode(rand(Float32, 10, 2))
x = ProductNode((pn, an3))
m = reflectinmodel(x)

@testset "list_lens" begin
    ls = list_lens(x)
    all_nodes = NodeIterator(x) |> collect
    all_fields = vcat(all_nodes, [md, an1.data, b.bags, an2.data, wb.bags, wb.weights, an3.data])
    all_fields = vcat(all_fields, Mill.metadata.(all_nodes))

    @test all(l -> only(getall(x, l)) in all_fields, ls)
    @test all(n -> n in all_fields, [walk(x, t) for t in list_traversal(x)])

    ls = list_lens(m)
    all_nodes = NodeIterator(m) |> collect
    all_fields = vcat(all_nodes, [m.m, m[1].m, m[1][:b].im.m, m[1][:b].a, m[1][:b].bm,
            m[1][:wb].im.m, m[1][:wb].a, m[1][:wb].bm, m[2].m])

    @test all(l -> only(getall(m, l)) in all_fields, ls)
    @test all(n -> n in all_fields, [walk(m, t) for t in list_traversal(m)])
end

@testset "findnonempty_lens" begin
    @test all(numobs.([only(getall(x, l)) for l in findnonempty_lens(x)]) .> 0)
end

@testset "find_lens" begin
    for t in list_traversal(x)
        ls = find_lens(x, x[t])
        @test all(l -> only(getall(x, l)) ≡ x[t], ls)
    end

    for t in list_traversal(m)
        ls = find_lens(m, m[t])
        @test all(l -> only(getall(m, l)) ≡ m[t], ls)
    end
end

@testset "code2lens & lens2code" begin
    for t in list_traversal(x)
        @test all(t .== vcat([lens2code(x, l) for l in code2lens(x, t)]...))
    end

    for t in list_traversal(m)
        @test all(t .≡ vcat([lens2code(m, l) for l in code2lens(m, t)]...))
    end
end

@testset "model_lens & data_lens" begin
    for n in NodeIterator(x)
        l = find_lens(x, n) |> only
        @test l == data_lens(x, model_lens(m, l))
    end
    for n in NodeIterator(m)
        l = find_lens(m, n) |> only
        @test l == model_lens(m, data_lens(x, l))
    end
end

@testset "replacein" begin
    for t in list_traversal(x)
        x2 = deepcopy(x)
        x3 = replacein(x2, x2[t], x[t])
        @test x3[t] ≡ x[t]
    end

    for t in list_traversal(m)
        m2 = deepcopy(m)
        m3 = replacein(m2, m2[t], m[t])
        @test m3[t] ≡ m[t]
    end
end
