# implements the multiple-instance learning model using Neural Networks, as described in
# https://arxiv.org/abs/1609.07257
# Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomas Pevny, Petr Somol
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated

# load the musk dataset
fMat = load("musk.jld2", "fMat")            # matrix with instances, each column is one sample
bagids = load("musk.jld2", "bagids")        # ties instances to bags
x = BagNode(ArrayNode(fMat), bagids)        # create BagDataset
y = load("musk.jld2", "y")                  # load labels
y = map(i -> maximum(y[i]) + 1, x.bags)     # create labels on bags
y_oh = Flux.onehotbatch(y, 1:2)             # one-hot encoding

# create the model
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.tanh)),                      # model on the level of Flows
    SegmentedMeanMax(10),                                       # aggregation
    ArrayModel(Chain(Dense(20, 10, Flux.tanh), Dense(10, 2))))  # model on the level of bags

# define loss function
loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

# the usual way of training
evalcb = throttle(() -> @show(loss(x, y_oh)), 1)
opt = Flux.ADAM()
@epochs 10 Flux.train!(loss, params(model), repeated((x, y_oh), 1000), opt, cb=evalcb)

# calculate the error on the training set (no testing set right now)
mean(mapslices(argmax, model(x).data, dims=1)' .!= y)
