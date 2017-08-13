
using GPUArrays
using BenchmarkTools

sizes = [x for x in 100:1000:10000];
cpu_times = Dict()
gpu_times = Dict()

println("\nCPU times")
for s in sizes
    X = rand(Float32,s,s);
    X_result = zeros(X);
    res_cpu = @elapsed A_mul_B!(X_result, X,X)
    println("size: ", s, " x ", s, " seconds: ", res_cpu, " seconds")
    #cpu_times[s] = mean(res_cpu.times)/10^6
end
