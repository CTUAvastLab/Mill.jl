using Documenter
using Mill

makedocs(
    sitename = "Mill",
    format = Documenter.HTML(),
    modules = [Mill]
)

deploydocs(
    repo = "github.com/pevnak/Mill.jl.git",
)
