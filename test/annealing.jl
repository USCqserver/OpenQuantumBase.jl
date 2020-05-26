using QTBase, Test

struct T_OPENSYS <: AbstractOpenSys end
struct T_COUPLINGS <: AbstractCouplings end

H = DenseHamiltonian([(x)->x], [σz])
u0 = PauliVec[1][1]

annealing = Annealing(H, u0)
@test annealing.H == H


ode_params = ODEParams(10)
@test ode_params.H == nothing
@test ode_params.tf == 10
ode_params = ODEParams(H, UnitTime(10), opensys = T_OPENSYS())
@test ode_params.H == H
@test ode_params.tf.t == 10
@test typeof(ode_params.opensys) == T_OPENSYS
