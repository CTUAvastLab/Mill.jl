using Revise
using NestedMill
using Base.Test
using Lazy

import NestedMill: DictEntry, recommendscheme, accomodate!, called

samples = open("prescription.jsonl") do fid
	readlines(fid)
end
schema = DictEntry()
foreach(f -> accomodate!(schema,JSON.parse(f)),samples)
reflector = NestedMill.recommendscheme(Float32,schema,2000)

println(NestedMill.tojson(reflector,2))
ds = @>> samples[1:1000] map(s-> reflector(JSON.parse(s)))
ds = cat(ds...)


