using Revise
using NestedMill
using Base.Test

import NestedMill: DictEntry

schema = DictEntry()
# schema = Dict{String,Any}()
open("prescription.jsonl") do fid
	foreach(f -> NestedMill.accomodate!(schema,JSON.parse(f)),readlines(fid))
end

