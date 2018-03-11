@begin testset 
	k = [2, 2, 2, 1, 1, 3]
	@test all(NestedMill.bag(k) .== [1:3,4:5,6:6])
end