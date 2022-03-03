using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated

using Random; Random.seed!(42);

fMat = load("musk.jld2", "fMat")

bagids = load("musk.jld2", "bagids")

y = load("musk.jld2", "y")

ds = BagNode(ArrayNode(fMat), bagids)

y = map(i -> maximum(y[i]) + 1, ds.bags)
y_oh = Flux.onehotbatch(y, 1:2)

model = BagModel(
    Dense(166, 10, Flux.tanh),
    BagCount(SegmentedMeanMax(10)),
    Chain(Dense(21, 10, Flux.tanh), Dense(10, 2)))

model(ds)

loss(ds, y_oh) = Flux.logitcrossentropy(model(ds), y_oh)

opt = Flux.ADAM()
@epochs 10 begin
    Flux.train!(loss, params(model), repeated((ds, y_oh), 1000), opt)
    println(loss(ds, y_oh))
end

mean(mapslices(argmax, model(ds), dims = 1)' .≠ y)

