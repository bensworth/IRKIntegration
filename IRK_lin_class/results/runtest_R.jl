methods = [
    # -3, -4,
    # 3, 4,
    # 23, 25, 27
    25
    # 32, 34, 36, 38
]

function mname(m)
    d = digits(m)
    if d[1] < 0
    	return "A-SDIRK $(abs(d[1]))"
    elseif length(d) == 1
        return "L-SDIRK $(d[1])"
    elseif d[end] == 1
        prefix = "Gauss"
    elseif d[end] == 2
        prefix = "Radau"
    elseif d[end] == 3
        prefix = "Lobatto"
    end

    order = sum([d[k]*10^(k-1) for k=1:length(d)-1])
    return string(prefix, " ", order)
end

for m in methods
    # for dt in [0.1, 0.05, 0.025, 0.0125, 0.00625]
    for dt in [0.025, 0.0125, 0.00625]
        cmd = `srun -n72 ./ord-red_dg_adv_diff -i $m -air 1 -rs 4 -rp 2 -o 4 -dt $dt -tf 2 -e 1e-4`
        # cmd = `srun -n72 ./ord-red_dg_adv_diff -i $m -air 1 -rs 4 -rp 0 -o 4 -dt $dt -tf 2 -e 1e-4`
        res = read(cmd, String)
        l2 = match(r"(?<=l2 )(.*?)(?=\n)", res)
        runtime = match(r"(?<=runtime )(.*?)(?=\n)", res)
        print(mname(m))
        println()
        println("dt ", dt)
        println("time ", runtime.match)
        println("l2 ", l2.match)
        println()
        flush(stdout)
    end
end
