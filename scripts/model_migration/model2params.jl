using Pkg

tempdir = mktempdir()
Pkg.activate(tempdir)

Pkg.add.(["JLD2", "FileIO", "Flux", PackageSpec(name="Mill", version="1.0.7")])
using Mill, Flux, JLD2, FileIO

# load your 1.0.7 model (via JLD2 or BSON)
# here I just define one
a = BagNode(ArrayNode(rand(3,4)),[1:4])
b = BagNode(ArrayNode(rand(3,4)),[1:4])
m = TreeNode((a,b))

ps = Flux.params(m)
@show ps
@save "ps.jld2" ps

