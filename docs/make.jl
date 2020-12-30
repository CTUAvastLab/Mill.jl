using Documenter, DocumenterCitations
using Mill

DocMeta.setdocmeta!(Mill, :DocTestSetup, :(using Mill); recursive=true)

makedocs(
         CitationBibliography(joinpath(@__DIR__, "references.bib")),
         sitename = "Mill.jl",
         format = Documenter.HTML(sidebar_sitename=false,
                                  prettyurls = get(ENV, "CI", nothing) == "true",
                                  assets = ["assets/favicon.ico", "assets/custom.css"]),
         modules = [Mill],
         pages = [
                  "Home" => "home.md",
                  "Motivation" => "motivation.md",
                  "Manual" => [
                                             "Nodes" => "manual/nodes.md",
                                             "Aggregations" => "manual/aggregation.md",
                                             "ReflectInModel" => "manual/reflectin.md",
                                             "Handling strings" => "manual/strings.md",
                                             "Missing values" => "manual/missing.md",
                                             "Custom Nodes" => "manual/custom.md"
                                            ],
                  "Examples" => [
                                 "Musk" => "examples/musk.md",
                                 "Advanced" => "examples/advanced.md",
                                 "GNN in 16 lines" => "examples/graphs.md",
                                 "DAGs" => "examples/dag.md"
                                ],
                  "Helper tools" => [
                                     "HierarichalUtils.jl" => "tools/hierarchical.md",
                                     # Conveniences (Lenses)
                                     # External Tools (JsonGrinder, HierarchicalUtils)
                                    ],
                  "References" => "references.md"
                 ],

        )

deploydocs(
           repo = "github.com/pevnak/Mill.jl.git",
          )
