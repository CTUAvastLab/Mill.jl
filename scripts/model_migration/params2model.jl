using Pkg

tempdir = mktempdir()
Pkg.activate(tempdir)

Pkg.add.(["Mill", "JLD2", "FileIO", "Flux"])
using Mill, Flux, JLD2, FileIO

# define a structure of the model (by hand or using a schema)
a = BagModel(ArrayModel(rand(3,4)), SegmentedMean(4), ArrayModel(rand(4, 2)))
b = BagModel(ArrayModel(rand(2,5)), SegmentedMean(5), ArrayModel(rand(2, 5)))
m = ProductModel((a,b))

@load "ps.jld2" ps
Flux.loadparams!(m, ps)
@show Flux.params(m)

# serialize/save your model again
# ...
