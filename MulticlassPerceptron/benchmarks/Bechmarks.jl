using MNIST
using BenchmarkTools

source_path = join(push!(split(pwd(),"/")[1:end-1],"source/" ),"/")

if !contains(==,LOAD_PATH, source_path)
    push!(LOAD_PATH, source_path)
end

using MulticlassPerceptron5
using MulticlassPerceptron4
using MulticlassPerceptron3
using MulticlassPerceptron2
using MulticlassPerceptron

percep1 = MulticlassPerceptron.MPerceptron(10,784)
percep2 = MulticlassPerceptron2.MPerceptron(10,784)
percep3 = MulticlassPerceptron3.MPerceptron(10,784)
percep4 = MulticlassPerceptron4.MPerceptron(10,784)
percep5 = MulticlassPerceptron5.MPerceptron(10,784)

n_classes = 10
n_features = 784
T = Float32

X_train, y_train = MNIST.traindata();
X_test, y_test = MNIST.testdata();
y_train = y_train + 1
y_test = y_test + 1;

T = Float32
X_train = Array{T}((X_train - minimum(X_train))/(maximum(X_train) - minimum(X_train)))
y_train = Array{Int64}(y_train)
X_test = Array{T}(X_test - minimum(X_test))/(maximum(X_test) - minimum(X_test))
y_test = Array{Int64}(y_test);


@benchmark MulticlassPerceptron.fit!(percep1, X_train, y_train, 1, 0.0001)
@benchmark MulticlassPerceptron2.fit!(percep2, X_train, y_train, 1, 0.0001)
@benchmark MulticlassPerceptron3.fit!(percep3, X_train, y_train, 1, 0.0001)
@benchmark MulticlassPerceptron4.fit!(percep4, X_train, y_train, 1, 0.0001)
@benchmark MulticlassPerceptron5.fit!(percep5, X_train, y_train, 1, 0.0001)
