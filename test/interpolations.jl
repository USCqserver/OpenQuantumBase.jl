using OpenQuantumBase, Test

x = range(0,stop=10,length=100)
y1 = Array(x) + 1.0im*Array(x)
y2 = (10.0+10.0im) .- Array(x)
inter_complex = construct_interpolations(x, y1)
inter_real = construct_interpolations(x, 10 .- collect(x))
gridded_2_range_inter = construct_interpolations(collect(x), y1)

test_x = range(0,stop=10,length=50)
exp_complex = collect(test_x) + 1.0im*collect(test_x)
res_complex = inter_complex.(test_x)
exp_real = (10.0) .- collect(test_x)
res_real = inter_real.(test_x)

@test res_complex ≈ exp_complex
@test res_real ≈ exp_real
@test gridded_2_range_inter.(test_x) ≈ exp_complex
@test gradient(inter_real, 2.3) ≈ -1
@test gradient(inter_real, [0.13, 0.21]) ≈ [-1, -1]

# Test for multi-demension array
y_array = transpose(hcat(y1, y2))
y_array_itp = construct_interpolations(x, y_array, order=1)
@test exp_complex ≈ y_array_itp(1, test_x)

# Test for extrapolation
x = range(1.0,stop=10.0)
y = 10 .- collect(x)
y_c = x + 1.0im*y

eitp = construct_interpolations(x, Array(x); extrapolation="line")
@test isapprox(eitp(0.0),0.0, atol=1e-8)
eitp = construct_interpolations(x, y_c; extrapolation="line")
@test imag(eitp(0)) ≈ 10
eitp = construct_interpolations(x, y_c; extrapolation="flat")
@test eitp(0) == 1.0 + 9.0im

x = [1.0, 3, 4, 5, 6, 7, 8, 9, 10]
@test_logs (:warn,"The grid is not uniform. Using grided linear interpolation.") construct_interpolations(x, 10 .- x, method="bspline")
