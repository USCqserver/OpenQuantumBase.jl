using QTBase, Test

struct T_OPENSYS <: QTBase.AbstractLiouvillian end
struct T_COUPLINGS <: QTBase.AbstractCouplings end
struct T_BATH <: QTBase.AbstractBath end

H = DenseHamiltonian([(x) -> x], [σz])
u0 = PauliVec[1][1]

annealing = Annealing(H, u0)
@test annealing.H == H
@test annealing.annealing_parameter(10, 5) == 0.5

evo = Evolution(H, u0)
@test evo.H == H
@test evo.annealing_parameter(10, 5) == 0.5


ode_params = ODEParams(T_OPENSYS(), 10, (tf, t)->t / tf)
@test typeof(ode_params.L) == T_OPENSYS
@test ode_params.tf == 10
@test ode_params(5) == 0.5

coupling = ConstantCouplings(["Z"])
inter = Interaction(coupling, T_BATH())
inter_set = InteractionSet(inter, inter)
@test inter_set[1] == inter
annealing = Annealing(H, u0, coupling = coupling, bath = T_BATH())
@test annealing.interactions[1].coupling == coupling
@test typeof(annealing.interactions[1].bath) <: T_BATH
@test_throws ArgumentError Annealing(H, u0, coupling = coupling, bath = T_BATH(), interactions = inter_set)
