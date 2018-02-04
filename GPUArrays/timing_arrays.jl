
using GPUArrays
using CLArrays
using BenchmarkTools

sizes = [x for x in 100:200:1000];
cpu_times = Dict()
gpu_times = Dict()

println("\nCPU times")
for s in sizes
    X = rand(Float32,s,s);
    X_result = zeros(X);
    res_cpu = @elapsed A_mul_B!(X_result, X, X)
    println("size: ", s, " x ", s, " seconds: ", res_cpu, " seconds")
    #println("size: ", s, " x ", s, " seconds: ", mean(res_cpu.times)/10^6, " seconds")
end


println("\nGPU times")
for s in sizes
    X = rand(Float32,s,s);
    X_result = zeros(X);
    X_gpu = CLArray(X);
    X_result_gpu = CLArray(zeros(Float32,s,s));

    res_gpu = @elapsed A_mul_B!(X_result_gpu, X_gpu, X_gpu)
    println("size: ", s, " x ", s, " seconds: ", res_gpu, " seconds")

    #println("size: ", s, " x ", s, " seconds: ", mean(res_gpu.times)/10^6, " seconds")
end


println("\nGPU times adding time to bring the array to the CPU")
for s in sizes
    X = rand(Float32,s,s);
    X_result = zeros(X);
    X_gpu = CLArray(X);
    X_result_gpu = CLArray(zeros(Float32,s,s));

    res_gpu = @elapsed synchronize(A_mul_B!(X_result_gpu, X_gpu, X_gpu))
    println("size: ", s, " x ", s, " seconds: ", res_gpu, " seconds")

    #println("size: ", s, " x ", s, " seconds: ", mean(res_gpu.times)/10^6, " seconds")

end
