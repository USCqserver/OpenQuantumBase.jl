"""
$(TYPEDEF)

Defines interpolating DenseHamiltonian object

# Fields

$(FIELDS)
"""
struct InterpDenseHamiltonian
    """Interpolating object"""
    interp_obj
    """Size"""
    size
end


"""
$(TYPEDEF)

Defines interpolating SparseHamiltonian object

# Fields

$(FIELDS)
"""
struct InterpSparseHamiltonian
    """Interpolating object"""
    interp_obj
    """Size"""
    size
end
