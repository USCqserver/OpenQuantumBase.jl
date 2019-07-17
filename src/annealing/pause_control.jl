function pausing_annealing_parameter(sp, sd)
    sf = 1 + sd
    stops = [sp, sp+sd]
    annealing_parameter = [(x)->x ,(x)->sp, (x)->x-sp]
    geometric_scaling = [1.0, 0.0, 1.0]
    AdiabaticFramePiecewiseControl(1.0, sf, stops, annealing_parameter, geometric_scaling)
end
