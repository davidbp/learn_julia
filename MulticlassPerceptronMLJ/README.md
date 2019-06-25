# Perceptron

Package for training Multiclass Perceptron models.

A library for those who feel deeply vanished from the (maybe non linearly separable) world.

A library for the deepfolks who, maybe, stumble upon linearly separable problems.

### Installation

You can clone the package
```julia
using Pkg
Pkg.clone("https://github.com/davidbp/MulticlassPerceptron.jl") 
```
Or you can add the package. Remember to be in pkg mode inside Julia (type `]`).
```
(v1.1) pkg> add "https://github.com/davidbp/MulticlassPerceptron.jl"
```


### Basic usage
```julia
using MLDatasets

train_x, train_y = MLDatasets.MNIST.traindata();
test_x, test_y = MLDatasets.MNIST.testdata();
train_x = Float32.(train_x);
test_x  = Float32.(test_x);
train_y = train_y .+ 1;
test_y  = test_y .+ 1;
train_x = reshape(train_x, 784, 60000);
test_x  = reshape(test_x,  784, 10000);
```

We can create a `PerceptronClassifer` type defining the type of the weights, the number of classes,
and the number of features.

The function `Perceptron.fit!` is used to train the model.

```julia
using MulticlassPerceptron
scores = []
n_features = size(train_x, 1);
n_classes =  length(unique(train_y));
perceptron = MulticlassPerceptronClassifier(Float32, n_classes, n_features);
MulticlassPerceptron.fit!(perceptron, train_x[:,1:10], train_y[1:10], scores;  print_flag=false, n_epochs=10);
```

#### Details of the `fit!` function

>    fit!(h::PerceptronClassifer,
>         X::Array,
>         y::Array;
>         n_epochs=50,
>         learning_rate=0.1,
>         print_flag=false,
>         compute_accuracy=true,
>         seed=srand(1234),
>         pocket=false,
>         shuffle_data=false)

##### Arguments

- **`h`**, (PerceptronClassifer{T} type), Multiclass perceptron.
- **`X`**, (Array{T,2} type), data contained in the columns of X.
- **`y`**, (Vector{T} type), class labels (as integers from 1 to n_classes).

##### Keyword arguments

- **`n_epochs`**, (Integer type), number of passes (epochs) through the data.
- **`learning_rate`**, (Float type), learning rate (The standard perceptron is with learning_rate=1.)
- **`compute_accuracy`**, (Bool type), if `true` the accuracy is computed at the end of every epoch.
- **`print_flag`**, (Bool type), if `true` the accuracy is printed at the end of every epoch.
- **`seed`**, (MersenneTwister type), seed for the permutation of the datapoints in case there the data is shuffled.
- **`pocket`** , (Bool type), if `true` the best weights are saved (in the pocket) during learning.
- **`shuffle_data`**, (Bool type),  if `true` the data is shuffled at every epoch (in reality we only shuffle indicies for performance).



#### Ascension from above: THe History of the profet who saw the light from the higher dimensions

The savant circle, ruler of flatland, told the triangle that it was impossible to cross the line river.
It was simply too long, far beyond the end of the realm the river went. One day, the stubborn triangle heard a voice: "the river can be crossed from above". What the hell is above? though the triangle.  The triangle tried to explain to the other peasants what was the world from above but nobody listened.

"This smartass thinks he can invent a word and sell us out on his dream. What on earth is `above` eh? show us"

The poor triangle asked the others to have faith and investigate other ways to solve the problems they faced but nobody was there. Everybody was too deeply focused on other stuff.


The story will continue ...
