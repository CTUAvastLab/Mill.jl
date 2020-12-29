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
         pages = ["Home" => ["index.md"],
                  "Examples" => [
                                 "Simple" => "examples/simple.md",
                                 "Advanced" => "examples/advanced.md",
                                 "GNN in 16 lines" => "examples/graphs.md",
                                 "DAGs" => "examples/dag.md",
                                ],
                  "Architecture of Mill" => [
                                             "Overview" => "architecture/overview.md",
                                             "ReflectInModel" => "architecture/reflectin.md",
                                             "Handling strings" => "architecture/strings.md",
                                             "Aggregations" => "architecture/aggregation.md",
                                             "Missing values" => "architecture/missing.md",
                                             "Custom Nodes" => "architecture/custom.md",
                                            ],
                  "Helper tools" => [
                                     "HierarichalUtils.jl" => "tools/hierarchical.md",
                                     # Conveniences (Lenses)
                                     # External Tools (JsonGrinder, HierarchicalUtils)
                                    ],
                 ],

        )

deploydocs(
           repo = "github.com/pevnak/Mill.jl.git",
          )
