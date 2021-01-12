using DelimitedFiles

dt = "2e-2"
stats = readdlm("solver_stats_dt_$dt.txt", Any)

rkmethods = [14, 23, 25, 34]

for m ∈ rkmethods
    m_stats = stats[stats[:,1] .== m, [2,7,3,4,5]]
    display(m_stats)
    writedlm("data_dt_$dt/m_$(m).txt", m_stats)
end
