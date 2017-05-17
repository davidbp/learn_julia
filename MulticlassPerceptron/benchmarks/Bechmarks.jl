using MNIST
using BenchmarkTools

source_path = join(push!(split(pwd(),"/")[1:end-1],"source/" ),"/")

if !contains(==,LOAD_PATH, source_path)
    push!(LOAD_PATH, source_path)
end

T = Float32

println("Starting Benchmark")

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

println("Scaling MNIST")

X_train = Array{T}((X_train - minimum(X_train))/(maximum(X_train) - minimum(X_train)))
y_train = Array{Int64}(y_train)
X_test = Array{T}(X_test - minimum(X_test))/(maximum(X_test) - minimum(X_test))
y_test = Array{Int64}(y_test);

println("\n\tPrecompining and timing")

@time MulticlassPerceptron1.fit!(percep1, X_train, y_train, 1, 0.0001)
@time MulticlassPerceptron2.fit!(percep2, X_train, y_train, 1, 0.0001)
@time MulticlassPerceptron3.fit!(percep3, X_train, y_train, 1, 0.0001)
@time MulticlassPerceptron4.fit!(percep4, X_train, y_train, 1, 0.0001)

println("\n\tReal timming")

@time MulticlassPerceptron1.fit!(percep1, X_train, y_train, 1, 0.0001)
@time MulticlassPerceptron2.fit!(percep2, X_train, y_train, 1, 0.0001)
@time MulticlassPerceptron3.fit!(percep3, X_train, y_train, 1, 0.0001)
@time MulticlassPerceptron4.fit!(percep4, X_train, y_train, 1, 0.0001)

