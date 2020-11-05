using Documenter
using Mill

makedocs(
    sitename = "Mill",
    format = Documenter.HTML(),
    modules = [Mill],
    pages = ["Home" => "index.md",
# Introduction to MIL, how it is different from GNN, what it is , and walktrough example
# Advanced example mixing products and bags (suggestions welcomed)
# Architecture of the library, parallels between Nodes and Models and minimum required functions for integration
# Handling missing values
# Performance Primitives (NGramStrings)
# Conveniences (Lenses)
# External Tools (JsonGrinder, HierarchicalUtils)
Suggested by Jan Francu, ?extensions to Graph NN? / Dags?
	],

)

deploydocs(
    repo = "github.com/pevnak/Mill.jl.git",
)
