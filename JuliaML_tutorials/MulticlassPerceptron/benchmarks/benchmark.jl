using BenchmarkTools
using MLDatasets
source_path = join(push!(split(pwd(),"/")[1:end-1],"source/" ),"/")
using MulticlassPerceptron

if !contains(==,LOAD_PATH, source_path)
    push!(LOAD_PATH, source_path)
end

T = Float32
println("Starting Benchmark")

begin
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

    println("\n\tMachine information")
    println("\n\tPeakflops:", peakflops())
    println("\n\tversioninfo:\n")
    versioninfo()
    println("\n\tComputing benchmarks:\n\n")
    println("\n\tTime to train 1 epoch MNIST:\n\n")
    results1 = @benchmark MulticlassPerceptron.fit!(percep1, X_train, y_train; n_epochs=1, print_flag=false, compute_accuracy=false)

    display(results1)
    println("\n")

    println("\n\tTraining 40 epochs MNIST:\n\n")
    @time MulticlassPerceptron.fit!(percep1, X_train, y_train;  n_epochs=40, print_flag=true, compute_accuracy=false)
    println("\n")

end
