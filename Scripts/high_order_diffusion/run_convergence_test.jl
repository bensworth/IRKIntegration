using DataFrames, Printf

orders = 2:10
methods = Dict(
   1 => [01],
   2 => [#= 02, =# 12, #= 32 =#],
   3 => [#= 03, =# 23],
   4 => [#= 04, =# 14, #= 34 =#],
   5 => [25],
   6 => [16, #= 36 =#],
   7 => [27],
   8 => [18, #= 38 =#],
   9 => [29],
   10 => [110]
)

method_names = Dict(
   01 => "BE", 02 => "DIRK2", 03 => "DIRK3", 04 => "DIRK4",
   12 => "Gauss2", 14 => "Gauss4", 16 => "Gauss6", 18 => "Gauss8", 110 => "Gauss10",
   23 => "Radau3", 25 => "Radau5", 27 => "Radau7", 29 => "Radau9",
   32 => "Lobatto2", 34 => "Lobatto4", 36 => "Lobatto6", 38 => "Lobatto8"
)

refs = 0:2

function run_convergence_tests()
   results = []
   for order ∈ orders
      for m ∈ methods[order]
         method = @sprintf "%02d" m
         for ref ∈ refs
            tf = 0.1
            dt = tf/2^ref

            cmd = `./heat -i $method -tf $tf -r $ref -dt $dt -o $(order-1)`
            println(cmd)
            s = read(cmd, String)
            er = parse(Float64, match(r"L2 error: (.*)", s).captures[1])
            println(er)

            current_result = Dict(
               "ref" => ref,
               "dt" => dt,
               "order" => order,
               "method_id" => m,
               "method" => method_names[m],
               "error" => er
            )
            push!(results, current_result)
         end
      end
   end
   df = vcat(DataFrame.(results)...)
end

println()

for order ∈ orders, m ∈ methods[order]
   errs = df[df.method_id .== m,:error]
   for i=1:(length(errs)-1)
      rate = log2(errs[i]/errs[i+1])
      @printf "%10s rate: %5.2f   (expected %d)\n" method_names[m] rate order
   end
end

function output_results(df)
   open("results.txt", write=true) do f
      print(f, "# ref\tdt")
      for order ∈ orders, m ∈ methods[order]
         print(f, "\t", method_names[methods[order][1]])
      end
      println(f)
      for ref ∈ refs
         print(f, ref, "\t", 0.1/2^ref)
         for order ∈ orders, m ∈ methods[order]
            print(f, "\t", df[(df.method_id .== m) .& (df.ref .== ref), :error][1])
         end
         println(f)
      end
   end
end
