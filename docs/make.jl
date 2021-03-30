using Documenter, DocumenterCitations
using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils

#=
Useful resources for writing docs:
    Julia guidelines: https://docs.julialang.org/en/v1/manual/documentation/
    Documenter syntax: https://juliadocs.github.io/Documenter.jl/stable/man/syntax/ 
    Showcase: https://juliadocs.github.io/Documenter.jl/stable/showcase/
    Doctests: https://juliadocs.github.io/Documenter.jl/stable/man/doctests/
=#

DocMeta.setdocmeta!(Mill, :DocTestSetup, quote
    using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils
end; recursive=true)

makedocs(
         CitationBibliography(joinpath(@__DIR__, "references.bib")),
         sitename = "Mill.jl",
         format = Documenter.HTML(sidebar_sitename=false,
                                  collapselevel = 1,
                                  prettyurls=get(ENV, "CI", nothing) == "true",
                                  assets=["assets/favicon.ico", "assets/custom.css"]),
         modules = [Mill],
         pages = [
                  "Home" => "index.md",
                  "Motivation" => "motivation.md",
                  "Manual" => [
                               "Nodes" => "manual/nodes.md",
                               "More on nodes" => "manual/more_on_nodes.md",
                               "Model reflection" => "manual/reflectin.md",
                               "Bag aggregation" => "manual/aggregation.md",
                               "Data in leaves" => "manual/leaf_data.md",
                               "Missing data" => "manual/missing.md",
                               "Custom nodes" => "manual/custom.md"
                              ],
                  "Examples" => [
                                 "Musk" => "examples/musk/musk.md",
                                 "GNNs in 16 lines" => "examples/graphs.md",
                                 "DAGs" => "examples/dag.md",
                                 "Processing JSONs" => "examples/jsons.md"
                                ],
                  "External tools" => [
                                       "HierarchicalUtils.jl" => "tools/hierarchical.md"
                                      ],
                  "Public API" => [
                            "Aggregation" => "api/aggregation.md",
                            "Bags" => "api/bags.md",
                            "Data nodes" => "api/data_nodes.md",
                            "Model nodes" => "api/model_nodes.md",
                            "Special Arrays" => "api/special_arrays.md",
                            "Switches" => "api/switches.md",
                            "Utilities" => "api/utilities.md"
                           ],
                  "References" => "references.md",
                  "Citation" => "citation.md"
                 ],

        )

deploydocs(
           repo = "github.com/CTUAvastLab/Mill.jl.git",
          )
