using Mill, Flux
using Distributions

function sample_ds1(n = 128)
    d1 = Binomial(40, 0.5)
    d2 = Binomial(40, 0.1)
    y = rand(1:2, n)
    bs = [rand(y[i] == 1 ? d1 : d2) for i in 1:n]
    BagNode(rand(2, sum(bs)) |> ArrayNode, length2bags(bs)), Flux.onehotbatch(y, 1:2)
end

function sample_ds2(n = 128)
    d1 = Binomial(80, 0.5)
    d2 = MixtureModel(Binomial[Binomial(80, 0.95), Binomial(80, 0.1)])
    y = rand(1:2, n)
    bs = [rand(y[i] == 1 ? d1 : d2) for i in 1:n]
    BagNode(rand(2, sum(bs)) |> ArrayNode, length2bags(bs)), Flux.onehotbatch(y, 1:2)
end

ds = sample_ds2
m = reflectinmodel(ds(1)[1]; b=Dict("" => k -> Chain(Dense(k, 10, relu), Dense(10, 2, relu))))
ps = Flux.params(m)
opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(m(x).data, y)
acc(x, y) = mean(Flux.onecold(m(x).data) .== Flux.onecold(y))

Flux.@epochs 10 begin
    Flux.train!(loss, ps, [ds() for _ in 1:1000], opt)
    val = ds()
    println(acc(val...))
end


