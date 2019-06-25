module MulticlassPerceptron

#using MetadataTools,  DocStringExtensions
using Random: shuffle, MersenneTwister

export MulticlassPerceptronClassifier, fit!, predict
using LinearAlgebra: mul!

#mutable struct MulticlassPerceptronClassifier{T}
#    W::AbstractMatrix{T}
#    b::AbstractVector{T}
#    n_classes::Int
#    n_features::Int
#end

mutable struct MulticlassPerceptronClassifier <: MLJBase.Deterministic
    weight_average::Bool
end

MLJBase.update(model::MulticlassPerceptronClassifier, verbosity, old_fitresult, old_cache, X, y) =
   MLJBase.fit(MulticlassPerceptronClassifier, verbosity, X, y)


# We should be able to pass an argument stating whether we want to save the evolution
# of the learning loss or not.                                                          
MLJBase.input_scitype_union(::Type{<:MulticlassPerceptronClassifier}) = Found




# fit returns coefficients minimizing a penalized rms loss function:
function MLJBase.fit(model::MulticlassPerceptronClassifier, X, y)
    x = MLJBase.matrix(X)                     # convert table to matrix
    #fitresult = (x'x - model.lambda*I)\(x'y)  # the coefficients
    
    fit!(model, X, y; model.n_epochs, model.learning_rate, model.print_flag,
         model.compute_accuracy, model.seed, model.pocket,
         model.shuffle_data)

    return fitresult
end

MulticlassPerceptronClassifier(; weight_average=false) = MulticlassPerceptronClassifier(weight_average)



function Base.show(io::IO, p::MulticlassPerceptronClassifier{T}) where T<:Number
    n_classes  = p.n_classes
    n_features = p.n_features

    print(io, "MulticlassPerceptronClassifier{$T}(n_classes=$n_classes, n_features=$n_features)")
end

MulticlassPerceptronClassifier(T::Type, n_classes::Int, n_features::Int) = MulticlassPerceptronClassifier{T}(rand(T, n_features, n_classes),
                                                                                       zeros(T, n_classes),
                                                                                       n_classes,
                                                                                       n_features)

"""
Compute the accuracy betwwen `y` and `y_hat`.
"""
function accuracy(y::AbstractVector, y_hat::AbstractVector)
    acc = 0.
    @fastmath for k = 1:length(y)
            @inbounds  acc += y[k] == y_hat[k]
    end
    return acc/length(y_hat)
end

"""
Predicts the class for a given input in a `MulticlassPerceptronClassifier`.
The placeholder is used to avoid allocating memory for each matrix-vector multiplication.

- Returns the predicted class.
"""
function predict(h::MulticlassPerceptronClassifier, x::AbstractVector, class_placeholder::AbstractVector)
    #@fastmath class_placeholder .= At_mul_B!(class_placeholder, h.W, x) .+ h.b
    class_placeholder .= mul!(class_placeholder, transpose(h.W), x)  .+ h.b
    return argmax(class_placeholder)
end

"""
Function to predict the class for a given example.

- Returns the predicted class.
"""
function predict(h::MulticlassPerceptronClassifier, x::AbstractVector)
    score = h.W' * x .+ h.b
    return argmax(score), score
end

"""
Function to predict the class for a given input batch.
- Returns the predicted class.
"""
function predict(h::MulticlassPerceptronClassifier, X::AbstractMatrix)
    predictions = zeros(Int64, size(X, 2))
    class_placeholder = zeros(eltype(h.W), h.n_classes)

    @inbounds for m in 1:length(predictions)
        predictions[m] = predict(h, view(X,:,m), class_placeholder)
    end
    return predictions
end

"""
>    fit!(h::MulticlassPerceptronClassifier,
>         X::Array,
>         y::Array;
>         n_epochs=50,
>         learning_rate=0.1,
>         print_flag=false,
>         compute_accuracy=true,
>         seed=Random.seed!(1234),
>         pocket=false,
>         shuffle_data=false)

##### Arguments

- **`h`**, (MulticlassPerceptronClassifier{T} type), Multiclass perceptron.
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
"""
function fit!(h::MulticlassPerceptronClassifier, X::AbstractArray, y::AbstractVector, scores::Array;
              n_epochs=50, learning_rate=1., print_flag=false,
              compute_accuracy=false, seed=MersenneTwister(1234), pocket=false,
              shuffle_data=false)

    n_features, n_samples = size(X)
    @assert length(y) == n_samples

    T = eltype(X)
    learning_rate     = T(learning_rate)
    class_placeholder = zeros(T, h.n_classes)
    y_preds           = zeros(Int64, n_samples)
    data_indices      = Array(1:n_samples)
    max_acc           = zero(T)

    if pocket
        W_hat = zeros(T, h.n_features, h.n_classes)
        b_hat = zeros(T, h.n_classes)
    end

    @fastmath for epoch in 1:n_epochs

        n_mistakes = 0
        if shuffle_data
            shuffle!(seed, data_indices)
        end
        #println("\nepoch ",epoch,"\n")
        @inbounds for m in data_indices
            #println("sample seen ", m ,"\n")
            x = view(X, :, m);
            y_hat = predict(h, x, class_placeholder)
            if y[m] != y_hat
                n_mistakes += 1
                ####  wij ← wij − η (yj −tj) · xi
                h.W[:, y[m]]  .= h.W[:, y[m]]  .+ learning_rate .* x
                h.b[y[m]]      = h.b[y[m]]     + learning_rate
                h.W[:, y_hat] .= h.W[:, y_hat] .- learning_rate .* x
                h.b[y_hat]     = h.b[y_hat]    - learning_rate
            end
        end

        #println("FINISHED")

        if compute_accuracy
             @inbounds for m in  data_indices
                 y_preds[m] = predict(h, view(X, :, m), class_placeholder)
            end
            acc = accuracy(y, y_preds)
            push!(scores, acc)
        else
            acc = (n_samples - n_mistakes)/n_samples
            push!(scores, acc)
        end

        if pocket
            if acc > max_acc
                max_acc = acc
                copy!(W_hat, h.W)
                copy!(b_hat, h.b)
            end
        end

        if print_flag
            print("Epoch: $(epoch) \t Accuracy: $(round(acc; digits=3))\r")
            #flush(STDOUT)
        end
    end
end


end # module
