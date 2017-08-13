
using GPUArrays
using BenchmarkTools

sizes = [x for x in 100:100:400];
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



println("\nGPU times")
for s in sizes
    X = rand(Float32,s,s);
    X_result = zeros(X);
    X_gpu = GPUArray(X);
    X_result_gpu = GPUArray(zeros(Float32,s,s));

    res_gpu = @elapsed A_mul_B!(X_result_gpu, X_gpu, X_gpu)
    println("size: ", s, " x ", s, " seconds: ", mean(res_gpu)/10^6, " seconds")
    #gpu_times[s] = mean(res_gpu.times)/10^6
end
