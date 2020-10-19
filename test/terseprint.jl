using Mill

function buf_show(data; kwargs...)
    buf = IOBuffer()
	Base.show(buf, data; kwargs...)
	String(take!(buf))
end

@testset "testing obscure terseprint things" begin
    function experiment(ds::LazyNode{T}) where {T<:Symbol}
    	@show ds
    	@show T
    end
    a = methods(experiment)
    a_method = a.ms[1]
    b = getfield(a_method.sig, 2)
    c = getfield(b, 3)
    d = c[2]
    e = d.body

    t = UnionAll(TypeVar(:t), LazyNode)
    u = typeof(LazyNode(:oh_hi, ["Mark"]))

    orig_terse = Mill._terseprint[]

    v = UnionAll(TypeVar(:T, AbstractMillModel), BagModel)

    Mill.terseprint(true)
    @test occursin("(ds::LazyNode{…}) where T<:Symbol", buf_show(methods(experiment)))
    @test buf_show(t) == "LazyNode{…}"
    @test buf_show(u) == "LazyNode{…}"
    @test buf_show(d) == "LazyNode{…}"
    @test buf_show(e) == "LazyNode{…}"
    @test buf_show(v) == "BagModel{…}"

    # extremely weird behavior, see https://github.com/pevnak/Mill.jl/issues/45
	Mill.terseprint(false)
    @test_throws ErrorException startswith("(ds::LazyNode{T,D} where D) where T<:Symbol", buf_show(methods(experiment)))
    @test_broken buf_show(methods(experiment))
    @test buf_show(t) == "LazyNode"
    @test buf_show(u) == "LazyNode{:oh_hi,Array{String,1}}"
    @test_throws ErrorException buf_show(d)
    @test_broken buf_show(d) == "LazyNode{T<:Symbol,D}"
	@test_throws ErrorException buf_show(e)
    @test_broken buf_show(e) == "LazyNode{T<:Symbol,D}"
    @test buf_show(v) == "BagModel"

    Mill.terseprint(orig_terse)
end
