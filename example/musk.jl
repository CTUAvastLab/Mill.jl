# implements the multiple-instance learning model using Neural Networks, as described in
# https://arxiv.org/abs/1609.07257
# Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomas Pevny, Petr Somol
using FileIO
using Flux
using MLDataPattern
using Flux: throttle
using Mill: BagNode, ArrayNode, segmented_meanmax, AggregationModel, ChainModel, reflectinmodel


# load the musk dataset
function loaddata()
  fMat = load("musk.jld","fMat");               # matrix with instances, each column is one sample
  bagids = load("musk.jld","bagids");           # ties instances to bags
  data = BagNode(ArrayNode(fMat), bagids)      # create BagDataset
  y = load("musk.jld","y");                     # load labels
  y = map(i -> maximum(y[i]) + 1, data.bags)    # create labels on bags
  return(data, y)
end


#create the model
model = AggregationModel(
    ChainModel(Dense(166, 10, Flux.relu)),   # model on the level of Flows
    segmented_meanmax,                                    # simultaneous mean and maximum is an all-time favorite
    ChainModel(Chain(Dense(20, 10, Flux.relu), Dense(10,2))))         # model on the level of bags

#define loss function
loss(x,y) = Flux.logitcrossentropy(model(getobs(x)), Flux.onehotbatch(y,1:2));

# the usual way of training
data, y = loaddata()
dataset = RandomBatches((data,y),100, 2000)
evalcb = () -> @show(loss(data, y))
opt = Flux.ADAM(params(model))
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

 # calculate the error on the training set (no testing set right now)
mean(mapslices(indmax,model(data),1)' .!= y)
