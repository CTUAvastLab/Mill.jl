using Random; Random.seed!(42)

using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated

fMat = load("musk.jld2", "fMat")            # matrix with instances, each column is one sample
bagids = load("musk.jld2", "bagids")        # ties instances to bags
ds = BagNode(ArrayNode(fMat), bagids)       # create a BagNode dataset
y = load("musk.jld2", "y")                  # load labels
y = map(i -> maximum(y[i]) + 1, ds.bags)    # create labels on bags
y_oh = Flux.onehotbatch(y, 1:2)             # one-hot encoding

# create the model
model = BagModel(
    Dense(166, 10, Flux.tanh),                      # model on the level of Flows
    BagCount(SegmentedMeanMax(10)),                             # aggregation
    Chain(Dense(21, 10, Flux.tanh), Dense(10, 2)))  # model on the level of bags

# check forward pass
model(ds)

# define loss function
loss(ds, y_oh) = Flux.logitcrossentropy(model(ds), y_oh)

# the usual way of training
opt = Flux.ADAM()
@epochs 10 begin
    Flux.train!(loss, params(model), repeated((ds, y_oh), 1000), opt)
    println(loss(ds, y_oh))
end

# calculate the error on the training set (no testing set right now)
mean(mapslices(argmax, model(ds), dims=1)' .≠ y)
