
replacein(x, oldnode, newnode) = x
replacein(x::Tuple, oldnode, newnode) = tuple([replacein(m, oldnode, newnode) for m in x]...)
replacein(x::NamedTuple, oldnode, newnode) = (;[k => replacein(x[k], oldnode, newnode) for k in keys(x)]...)

for T in [ArrayNode, ArrayModel, BagModel, BagNode, TreeNode, ProductModel]
	@eval begin
		function replacein(x::$T, oldnode, newnode)
			x == oldnode && return(newnode)
			$T(map(f -> replacein(getproperty(x, f), oldnode, newnode), fieldnames($T))...)
		end
	end
end