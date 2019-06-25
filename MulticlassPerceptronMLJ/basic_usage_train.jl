
using MLDatasets, Statistics
using MulticlassPerceptron

## Prepare data
train_x, train_y = MLDatasets.MNIST.traindata();
test_x, test_y   = MLDatasets.MNIST.testdata();
train_x = Float32.(train_x);
test_x  = Float32.(test_x);
train_y = train_y .+ 1;
test_y  = test_y .+ 1;
train_x = reshape(train_x, 784, 60000);
test_x  = reshape(test_x,  784, 10000);

## Define model and train it
scores = []
n_features = size(train_x, 1);
n_classes =  length(unique(train_y));
perceptron = MulticlassPerceptronClassifier(Float32, n_classes, n_features);
println("\nStart Learning\n")
MulticlassPerceptron.fit!(perceptron, train_x, train_y, scores;  print_flag=true, n_epochs=100);
y_hat_test = MulticlassPerceptron.predict(perceptron, test_x);
y_hat_train = MulticlassPerceptron.predict(perceptron,train_x)
println("\nLearning Finished\n")

println("Results:")
println("Train accuracy:", mean(y_hat_train .==train_y))
println("Test accuracy:", mean(y_hat_test .==test_y))

