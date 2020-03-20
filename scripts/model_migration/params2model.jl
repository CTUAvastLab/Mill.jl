using Pkg

tempdir = mktempdir()
Pkg.activate(tempdir)

Pkg.add.(["Mill", "JLD2", "FileIO", "Flux"])
using Mill, Flux, JLD2, FileIO

# define a structure of the model (by hand or using a schema)
a = BagNode(ArrayNode(rand(3,4)),[1:4])
b = BagNode(ArrayNode(rand(3,4)),[1:4])
m = TreeNode((a,b))

@load "ps.jld2" ps
Flux.loadparams!(m, ps)
@show Flux.params(m)

# serialize/save your model again
# ...

