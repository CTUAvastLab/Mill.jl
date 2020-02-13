using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using CuArrays

CuArrays.allowscalar(false)

fMat = load("example/musk.jld2", "fMat");      # matrix with instances, each column is one sample
bagids = load("example/musk.jld2", "bagids");  # ties instances to bags
x = BagNode(ArrayNode(fMat), bagids)
y = load("example/musk.jld2", "y");            # load labels
y = map(i -> maximum(y[i]) + 1, x.bags);       # create labels on bags

x = x |> Mill.gpu
y_oh = Flux.onehotbatch(y, 1:2) |> Mill.gpu

model = BagModel(
    ArrayModel(Dense(166, 10, Flux.tanh)),                      # model on the level of Flows
    SegmentedMeanMax(10),                                       # aggregation
    ArrayModel(Chain(Dense(20, 10, Flux.tanh), Dense(10, 2)))) |> Mill.gpu

loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)
evalcb = () -> @show(loss(x, y_oh));

opt = Flux.ADAM();
@epochs 100 Flux.train!(loss, Flux.params(model), repeated((x, y_oh), 100), opt, cb=throttle(evalcb, 1))
