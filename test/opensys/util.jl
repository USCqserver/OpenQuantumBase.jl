using OpenQuantumBase

w = [-3, 1, 3, 3, 4.5, 5.5, 8]
gidx = OpenQuantumBase.GapIndices(w, 8, 8)
uniq_w, indices, indices0 = OpenQuantumBase.find_unique_gap(w)
@test gidx.uniq_w == uniq_w
@test gidx.uniq_a == [[x.I[1] for x in i] for i in indices]
@test gidx.uniq_b == [[x.I[2] for x in i] for i in indices]
@test Set([(i,j) for (i,j) in zip(gidx.a0, gidx.b0)]) == Set([(x.I[1], x.I[2]) for x in indices0])

