using QTBase, Test

struct T_OPENSYS <: AbstractOpenSys end
struct T_COUPLINGS <: AbstractCouplings end
struct T_BATH <: AbstractBath end

H = DenseHamiltonian([(x) -> x], [σz])
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

coupling = ConstantCouplings(["Z"])
inter = Interaction(coupling, T_BATH())
inter_set = InteractionSet(inter, inter)
@test inter_set[1] == inter
annealing = Annealing(H, u0, coupling = coupling, bath = T_BATH())
@test annealing.interactions[1].coupling == coupling
@test typeof(annealing.interactions[1].bath) <: T_BATH
@test_throws ArgumentError Annealing(H, u0, coupling = coupling, bath = T_BATH(), interactions = inter_set)
