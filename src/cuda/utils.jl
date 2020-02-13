"""
    Gets fully qualified UnionAll corresponding to T
"""
function get_unionall(T::DataType)
    m = T.name.module
    t = T.name.name

    :($(m).$(t))
end
