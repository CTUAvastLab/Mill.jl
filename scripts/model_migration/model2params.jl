using Pkg

tempdir = mktempdir()
Pkg.activate(tempdir)

Pkg.add.(["JLD2", "FileIO", "Flux", PackageSpec(name="Mill", version="1.0.7")])
using Mill, Flux, JLD2, FileIO

# load your 1.0.7 model (via JLD2 or BSON)
# here I just define one
a = BagModel(ArrayModel(rand(3,4)), SegmentedMean(4), ArrayModel(rand(4, 2)))
b = BagModel(ArrayModel(rand(2,5)), SegmentedMean(5), ArrayModel(rand(2, 5)))
m = ProductModel((a,b))

ps = Flux.params(m)
@show ps
@save "ps.jld2" ps

