using Documenter
using Mill

makedocs(
    sitename = "Mill",
    format = Documenter.HTML(),
    modules = [Mill]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
