using FileIO
using JLD2
using Flux
using MLDataPattern
using Flux: throttle, @epochs
using Mill
using Mill: reflectinmodel
using Statistics

function loaddata()
  fMat = load("example/musk.jld2","fMat");               # matrix with instances, each column is one sample
  bagids = load("example/musk.jld2","bagids");           # ties instances to bags
  data = BagNode(ArrayNode(fMat), bagids)      # create BagDataset
  y = load("example/musk.jld2","y");                     # load labels
  y = map(i -> maximum(y[i]) + 1, data.bags)    # create labels on bags
  return(data, y)
end
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.relu)),   # model on the level of Flows
    SegmentedMax(),
    ArrayModel(Chain(Dense(10, 10, Flux.relu), Dense(10, 2))))         # model on the level of bags
loss(x,y) = Flux.logitcrossentropy(model(getobs(x)).data, Flux.onehotbatch(y, 1:2));

data, y = loaddata()
dataset = RandomBatches((data,y), 100, 2000)
evalcb = () -> @show(loss(data, y))
opt = Flux.ADAM(params(model))

function profile_test(n)
    @epochs n Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))
end

profile_test(1)
using Profile
Profile.clear()
@profile profile_test(3)

using ProfileView
ProfileView.view()
