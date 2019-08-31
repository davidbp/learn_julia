using BenchmarkTools
using MLDatasets

source_path = join(push!(split(pwd(),"/")[1:end-1],"source/" ),"/")

if !contains(==,LOAD_PATH, source_path)
    push!(LOAD_PATH, source_path)
end
using MulticlassPerceptron


T = Float32
println("Starting Benchmark")

println("Loading Modules")
percep1 = MulticlassPerceptron.MPerceptron(T, 10, 784)
n_classes = 10
n_features = 784

println("Loading MNIST")
X_train, y_train = MLDatasets.MNIST.traindata();
X_test, y_test = MLDatasets.MNIST.MNIST.testdata();
X_train = reshape(X_train, (784,60000))
X_test = reshape(X_test, (784,10000))

y_train = y_train + 1
y_test = y_test + 1;
X_train = Array{T}((X_train - minimum(X_train))/(maximum(X_train) - minimum(X_train)))
y_train = Array{Int64}(y_train)
X_test = Array{T}(X_test - minimum(X_test))/(maximum(X_test) - minimum(X_test))
y_test = Array{Int64}(y_test);

@time fit!(percep1, X_train, y_train; 
	 n_epochs=10, print_flag=false, compute_accuracy=false)
