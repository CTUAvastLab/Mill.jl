using FileIO, JLD2, Statistics, Mill, Flux, OneHotArrays

using Random; Random.seed!(42);

fMat = load("musk.jld2", "fMat")

bagids = load("musk.jld2", "bagids")

y = load("musk.jld2", "y")

ds = BagNode(ArrayNode(fMat), bagids)

y = map(i -> maximum(y[i]) + 1, ds.bags)
y_oh = onehotbatch(y, 1:2)

model = BagModel(
    Dense(166, 50, Flux.tanh),
    SegmentedMeanMax(50),
    Chain(Dense(100, 50, Flux.tanh), Dense(50, 2)))

model(ds)

opt_state = Flux.setup(Adam(), model);

loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y);

for e in 1:100
    if e % 10 == 1
        @info "Epoch $e" training_loss=loss(model, ds, y_oh)
    end
    Flux.train!(loss, model, [(ds, y_oh)], opt_state)
end

mean(Flux.onecold(model(ds), 1:2) .== y)
