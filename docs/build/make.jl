using Documenter
using Mill

makedocs(
    sitename = "Mill",
    modules = [Mill],
    format = Documenter.HTML(
		assets = ["assets/flux.css"]),
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
