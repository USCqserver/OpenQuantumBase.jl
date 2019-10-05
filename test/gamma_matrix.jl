using QTBase, Test
#TODO The test currently covers only the 2-level case. Need to be extended to high dimension

t = range(0,1,length=100)
Γ = zeros(1,2,100)

Γ[1,1,:] = 1:100
Γ[1,2,:] = 100:-1:1

Γobj = ΓMatrix(t, Γ)

@test Γobj(1.0) == [-100 1; 100 -1]
@test Γobj(0.1) == [-10.9 90.1; 10.9 -90.1]
