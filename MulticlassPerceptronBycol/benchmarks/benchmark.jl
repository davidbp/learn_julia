using MNIST
using BenchmarkTools

source_path = join(push!(split(pwd(),"/")[1:end-1],"source/" ),"/")

if !contains(==,LOAD_PATH, source_path)
    push!(LOAD_PATH, source_path)
end

T = Float32

println("Starting Benchmark")
begin
using MulticlassPerceptron4
using MulticlassPerceptron3
using MulticlassPerceptron2
using MulticlassPerceptron1

println("Loading Modules")

percep1 = MulticlassPerceptron1.MPerceptron(T, 10, 784)
percep2 = MulticlassPerceptron2.MPerceptron(T, 10, 784)
percep3 = MulticlassPerceptron3.MPerceptron(T, 10, 784)
percep4 = MulticlassPerceptron4.MPerceptron(T, 10, 784)

n_classes = 10
n_features = 784

println("Loading MNIST")
X_train, y_train = MNIST.traindata();
X_test, y_test = MNIST.testdata();
y_train = y_train + 1
y_test = y_test + 1;
X_train = Array{T}((X_train - minimum(X_train))/(maximum(X_train) - minimum(X_train)))
y_train = Array{Int64}(y_train)
X_test = Array{T}(X_test - minimum(X_test))/(maximum(X_test) - minimum(X_test))
y_test = Array{Int64}(y_test);

println("\n\tMachine information")

println("\n\tPeakflops:", peakflops())

println("\n\tversioninfo:\n")
versioninfo()

println("\n\tComputing benchmarks... ")

results1 = @benchmark MulticlassPerceptron1.fit!(percep1, X_train, y_train, 1, 0.0001)
results2 = @benchmark MulticlassPerceptron2.fit!(percep2, X_train, y_train, 1, 0.0001)
results3 = @benchmark MulticlassPerceptron3.fit!(percep3, X_train, y_train, 1, 0.0001)
results4 = @benchmark MulticlassPerceptron4.fit!(percep4, X_train, y_train, 1, 0.0001)


println("\nPerceptron1---------------------")
display(results1)
println("\n\nPerceptron2---------------------")
display(results2)
println("\n\nPerceptron3---------------------")
display(results3)
println("\n\nPerceptron4---------------------")
display(results4)
println("\n")
end
