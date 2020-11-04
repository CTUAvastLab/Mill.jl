using Documenter
using Mill

makedocs(
    sitename = "Mill",
    modules = [Mill],
    pages = ["Home" => "index.md",
		# ["Basics" => "models/basics.md",
		#  "Recurrence" => "models/recurrence.md",
		#  "Model Reference" => "models/layers.md",
		#  "Loss Functions" => "models/losses.md",
		#  "Regularisation" => "models/regularisation.md",
		#  "Advanced Model Building" => "models/advanced.md",
		#  "NNlib" => "models/nnlib.md"],
		],
    format = Documenter.HTML(
		assets = ["assets/flux.css"]),
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
