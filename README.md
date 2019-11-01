# Mill – Multiple Instance Learning Library
 Mill is a library build on top of Flux.jl aimed to prototype flexible multi-instance learning models as described in  [[1](#cit1)] and  [[2](#cit2)]
 
 

 ## What is Multiple instance learning (MIL) problem?
 In the prototypical machine learning problem the input sample ![equation](https://latex.codecogs.com/gif.latex?x) is a vector or matrix of a fixed dimension, or a sequence. In MIL problems the sample ![equation](https://latex.codecogs.com/gif.latex?x) is a set of vectors (or matrices) ![equation](https://latex.codecogs.com/gif.latex?%28x_1%2C%20x_2%2C%20...%2C%20x_n%29), which means that order does not matter, and which is also the feature making MIL problems different from sequences.
 Pevny and Somol has proposed simple way to solve MIL problems with neural networks. The network consists from two non-linear layers, with mean (or maximum) operation sandwiched between nonlinearities. Denoting ![equation](https://latex.codecogs.com/gif.latex?f_1), ![equation](https://latex.codecogs.com/gif.latex?f_2) layers of neural network, the output is calculated as ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20f_2%20%5Cleft%28%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20f_1%28x_i%29%5Cright%29). In [[3](#cit3)], authors have further extended the universal approximation theorem to MIL problems.
 ### Multiple instance learning on Musk 1
Musk dataset is a classic problem of the field used in publication [[4](#cit4)], which has given the class of problems its name. 
 Below is a little walk-through how to solve the problem using Mill library. The full example is shown in [example/musk.jl](example/musk.jl).
 
 Let's start by importing all libraries
```julia
julia> using FileIO, JLD2, Statistics, Mill, Flux
julia> using Flux: throttle, @epochs
julia> using Mill: reflectinmodel
julia> using Base.Iterators: repeated
```
 Loading a dataset from file and folding it in Mill's data-structures is done in the following function. `musk.jld2` contains matrix with features, `fMat`, the id of sample (called bag in MIL terminology) to which each instance (column in `fMat`) belongs to, and finally a label of each instance in `y`. 
`BagNode` is a structure which holds feature matrix and ranges of columns of each bag. Finally, `BagNode` can be concatenated (use `catobs`) and you can get subset using `getindex`.
```julia
julia> fMat = load("example/musk.jld2", "fMat");      # matrix with instances, each column is one sample
julia> bagids = load("example/musk.jld2", "bagids");  # ties instances to bags
julia> x = BagNode(ArrayNode(fMat), bagids);          # create BagDataset
julia> y = load("example/musk.jld2", "y");            # load labels
julia> y = map(i -> maximum(y[i]) + 1, x.bags);       # create labels on bags
julia> y_oh = Flux.onehotbatch(y, 1:2);               # one-hot encoding
```
 Once we have data, we can manually create a model. `BagModel` is designed to implement a basic multi-instance learning model as described above. Below, we use a simple model, where instances are first passed through a single layer with 10 neurons (input dimension is 166) with `tanh` non-linearity, then we use `mean` and `max` aggregation functions simultaneously (for some problems, max is better then mean, therefore we use both), and then we use one layer with 10 neurons and `tanh` nonlinearity followed by output linear layer with 2 neurons (output dimension).
```julia
julia> model = BagModel(
    ArrayModel(Dense(166, 10, Flux.tanh)),                      # model on the level of Flows
    SegmentedMeanMax(10),                                       # aggregation
    ArrayModel(Chain(Dense(20, 10, Flux.tanh), Dense(10, 2))))  # model on the level of bags

BagModel
  ├── ArrayModel(Dense(166, 10, NNlib.tanh))
  ├── ⟨SegmentedMean(10), SegmentedMax(10)⟩
  └── ArrayModel(Chain(Dense(20, 10, NNlib.tanh), Dense(10, 2)))
```
 The loss function is standard `cross-entropy`:
```julia
julia> loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);
```
 Finally, we put everything together. The below code should resemble an example from `Flux.jl` library.
 ```julia
julia> evalcb = () -> @show(loss(x, y_oh));
julia> opt = Flux.ADAM();
julia> @epochs 10 Flux.train!(loss, params(model), repeated((x, y_oh), 100), opt, cb=throttle(evalcb, 1))

[ Info: Epoch 1
loss(x, y_oh) = 87.793724f0
[ Info: Epoch 2
loss(x, y_oh) = 4.3207192f0
[ Info: Epoch 3
loss(x, y_oh) = 4.2778687f0
[ Info: Epoch 4
loss(x, y_oh) = 0.662226f0
[ Info: Epoch 5
loss(x, y_oh) = 5.76351f-6
[ Info: Epoch 6
loss(x, y_oh) = 3.8146973f-6
[ Info: Epoch 7
loss(x, y_oh) = 2.8195589f-6
[ Info: Epoch 8
loss(x, y_oh) = 2.4878461f-6
[ Info: Epoch 9
loss(x, y_oh) = 2.1561332f-6
[ Info: Epoch 10
loss(x, y_oh) = 1.7414923f-6
```
 
Because we did not leave any data for validation, we can only calculate error on the training data, which should be not so surprisingly low.
 ```julia
mean(mapslices(argmax, model(x).data, dims=1)' .!= y)

0.0
```
 ### More complicated models
The main advantage of the Mill library is that it allows to arbitrarily nest and cross-product `BagModels`, as is described in Theorem 5 of [[3](#cit3)].
 Let's start the demonstration by nesting two MIL problems. The outer MIL model contains three samples. The first sample contains another bag (inner MIL) problem with two instances, the second sample contains two inner bags with total of three instances, and finally the third sample contains two inner bags with four instances.
```julia
julia> ds = BagNode(BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),[1:1,2:3,4:5])
BagNode with 3 bag(s)
  └── BagNode with 5 bag(s)
        └── ArrayNode(4, 10)
```
 We can create the model manually as in the case of Musk as
```julia
julia> m = BagModel(
    BagModel(
        ArrayModel(Dense(4, 3, Flux.relu)),   
        SegmentedMeanMax(3),
        ArrayModel(Dense(6, 3, Flux.relu))),
    SegmentedMeanMax(3),
    ArrayModel(Chain(Dense(6, 3, Flux.relu), Dense(3,2))))

BagModel
  ├── BagModel
  │     ├── ArrayModel(Dense(4, 3, NNlib.relu))
  │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  └── ArrayModel(Chain(Dense(6, 3, NNlib.relu), Dense(3, 2)))
```
and we can apply the model as
```julia
julia> m(ds)

ArrayNode(2, 3)
```
 Since constructions of large models can be a process prone to errors, there is a function `reflectinmodel` which tries to automatize it keeping track of dimensions. It accepts a first parameter a sample `ds`, the second is a function returning layer (or set of layers) with input dimension `d`, and the third function is a function returning aggregation functions for `BagModel`. Using the function on the above example creates a model:
```julia
julia> m = reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMeanMax(d))

BagModel
  ├── BagModel
  │     ├── ArrayModel(Dense(4, 5, NNlib.relu))
  │     ├── ⟨SegmentedMean(5), SegmentedMax(5)⟩
  │     └── ArrayModel(Dense(10, 5, NNlib.relu))
  ├── ⟨SegmentedMean(5), SegmentedMax(5)⟩
  └── ArrayModel(Dense(10, 5, NNlib.relu))
```
Let's test the model
```julia
julia> m(ds).data

5×3 Array{Float32,2}:
 0.0542484   0.733629  0.553823
 0.062246    0.866254  1.03062 
 0.027454    1.04703   1.63135 
 0.00796955  0.36415   1.18108 
 0.034735    0.17383   0.0
```
 ### Even more complicated models
As already mentioned above, the datasets can contain Cartesian products of MIL and normal (non-MIL) problems. Let's do a quick demo.
```julia
julia> ds = BagNode(
    TreeNode(
        (BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),
        ArrayNode(randn(3,5)),
        BagNode(
            BagNode(ArrayNode(randn(2,30)),[i:i+1 for i in 1:2:30]),
            [1:3,4:6,7:9,10:12,13:15]),
        ArrayNode(randn(2,5)))),
    [1:1,2:3,4:5])

BagNode with 3 bag(s)
  └── TreeNode
        ├── BagNode with 5 bag(s)
        │     └── ArrayNode(4, 10)
        ├── ArrayNode(3, 5)
        ├── BagNode with 5 bag(s)
        │     └── BagNode with 15 bag(s)
        │           └── ArrayNode(2, 30)
        └── ArrayNode(2, 5)
```
For this, we really want to create model automatically despite it being sub-optimal.
```julia
julia> m = reflectinmodel(ds, d -> Dense(d, 3, relu), d -> SegmentedMeanMax(d))

BagModel
  ├── ProductModel (
  │     ├── BagModel
  │     │     ├── ArrayModel(Dense(4, 3, NNlib.relu))
  │     │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  │     ├── ArrayModel(Dense(3, 3, NNlib.relu))
  │     ├── BagModel
  │     │     ├── BagModel
  │     │     │     ├── ArrayModel(Dense(2, 3, NNlib.relu))
  │     │     │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     │     │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  │     │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  │     └── ArrayModel(Dense(2, 3, NNlib.relu))
  │   ) ↦  ArrayModel(Dense(12, 3, NNlib.relu))
  ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  └── ArrayModel(Dense(6, 3, NNlib.relu))

```

## Tree traversals
The latest version also includes a convenient traversal functionality:
```julia
julia>  show_traversal(m)

BagModel []
  ├── ProductModel [W] (
  │     ├── BagModel [a]
  │     │     ├── ArrayModel(Dense(4, 3, NNlib.relu)) [c]
  │     │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  │     ├── ArrayModel(Dense(3, 3, NNlib.relu)) [e]
  │     ├── BagModel [i]
  │     │     ├── BagModel [k]
  │     │     │     ├── ArrayModel(Dense(2, 3, NNlib.relu)) [l]
  │     │     │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     │     │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  │     │     ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  │     │     └── ArrayModel(Dense(6, 3, NNlib.relu))
  │     └── ArrayModel(Dense(2, 3, NNlib.relu)) [m]
  │   ) ↦  ArrayModel(Dense(12, 3, NNlib.relu))
  ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  └── ArrayModel(Dense(6, 3, NNlib.relu))
```

This way any node in the model tree is swiftly accessible, which may come in handy when inspecting model parameters or simply deleting/replacing/inserting nodes to tree. All tree nodes are accessible by indexing with the traversal code:.

```julia
julia> m["k"]

BagModel
  ├── ArrayModel(Dense(2, 3, NNlib.relu))
  ├── ⟨SegmentedMean(3), SegmentedMax(3)⟩
  └── ArrayModel(Dense(6, 3, NNlib.relu))
```

The following two approaches give the same result:
```julia
julia> m["k"] === m.im.ms[3].im

true
```

## Default aggregation values
With the latest version of Mill, it is also possible to work with missing data, replacing a missing bag with a default constant value, and even to learn this value as well.

## References
 <a name="cit1"><b>1</b></a> *Discriminative models for multi-instance problems with tree-structure, Tomáš Pevný, Petr Somol, 2016*, https://arxiv.org/abs/1703.02868
 
 <a name="cit2"><b>2</b></a> *Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomáš Pevný, Petr Somol, 2016*, https://arxiv.org/abs/1609.07257. 
 
 <a name="cit3"><b>3</b></a> *Approximation capability of neural networks on sets of probability measures and tree-structured data, Tomáš Pevný, Vojtěch Kovařík, 2019*, https://openreview.net/forum?id=HklJV3A9Ym
 
 <a name="cit4"><b>4</b></a> *Solving the multiple instance problem with axis-parallel rectangles, Dietterich, Thomas G., Richard H. Lathrop, and Tomás Lozano-Pérez, 1997*
