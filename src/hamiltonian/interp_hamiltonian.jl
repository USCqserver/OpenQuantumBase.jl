"""
$(TYPEDEF)

Defines interpolating DenseHamiltonian object

# Fields

$(FIELDS)
"""
struct InterpDenseHamiltonian <: AbstractDenseHamiltonian
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
struct InterpSparseHamiltonian{grided} <:AbstractSparseHamiltonian
    """Interpolating object"""
    interp_obj
    """Size"""
    size
end


function InterpDenseHamiltonian(s, hmat; method = "bspline", order = 1)
    construct_interpolations(s, hmat)
end
