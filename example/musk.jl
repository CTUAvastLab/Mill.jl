# implements the multiple-instance learning model using Neural Networks, as described in
# https://arxiv.org/abs/1609.07257
# Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomas Pevny, Petr Somol
using FileIO
using JLD2
using Flux
using MLDataPattern
using Flux: throttle
using Mill
using Mill: reflectinmodel
using Statistics


# load the musk dataset
function loaddata()
  fMat = load("musk.jld2","fMat");               # matrix with instances, each column is one sample
  bagids = load("musk.jld2","bagids");           # ties instances to bags
  data = BagNode(ArrayNode(fMat), bagids)      # create BagDataset
  y = load("musk.jld2","y");                     # load labels
  y = map(i -> maximum(y[i]) + 1, data.bags)    # create labels on bags
  return(data, y)
end


#create the model
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.relu)),   # model on the level of instances
    SegmentedMeanMax(10),
    ArrayModel(Chain(Dense(20, 10, Flux.relu), Dense(10, 2))))         # model on the level of bags

#define loss function
loss(x,y) = Flux.logitcrossentropy(model(getobs(x)).data, Flux.onehotbatch(y, 1:2));

# the usual way of training
data, y = loaddata()
dataset = RandomBatches((data,y), 100, 10000)
evalcb = () -> @show(loss(data, y))
opt = Flux.ADAM()
Flux.train!(loss, params(model), dataset, opt, cb = throttle(evalcb, 1))

 # calculate the error on the training set (no testing set right now)
Statistics.mean(mapslices(argmax, model(data).data, dims=1)' .!= y)
