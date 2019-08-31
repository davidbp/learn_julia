
module MulticlassPerceptron
export predict, MPerceptron, fit!, accuracy, fit2!

type MPerceptron{T}
    W::AbstractMatrix
    b::AbstractVector{T}
    n_classes::Int
    n_features::Int
    accuracy::AbstractVector{T}
end

function Base.show{T}(io::IO, p::MPerceptron{T})
    n_classes  = p.n_classes
    n_features = p.n_features
    print(io, "Perceptron{$T}(n_classes=$n_classes, n_features=$n_features)")
end

MPerceptron(T::Type, n_classes::Int, n_features::Int) = MPerceptron(rand(T, n_features, n_classes),
                                                                    zeros(T, n_classes),
                                                                    n_classes,
                                                                    n_features,
                                                                    zeros(T,0))
"""
Compute the accuracy betwwen `y` and `y_hat`.
"""
function accuracy(y, y_hat)
    acc = 0.
    @inbounds for k = 1:length(y)
                 acc += y[k] == y_hat[k]
    end
    return acc/length(y_hat)
end

"""
Function to predict the class for a given input in a `MPerceptron`.
The placeholder is used to avoid allocating memory for each matrix-vector multiplication.

- Returns the predicted class.
"""
function predict(h::MPerceptron, x, placeholder)
    placeholder .= At_mul_B!(placeholder, h.W, x) .+ h.b
    return indmax(placeholder)
end

"""
Function to predict the class for a given input in a `MPerceptron`.

- Returns the predicted class.
"""
function predict(h::MPerceptron, x)
    return indmax(h.W' * x .+ h.b)
end

"""
>    fit!(h::Perceptron,
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

- **`h`**, (MPerceptron{T} type), initialized perceptron.
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
function fit!(h::MPerceptron, X::AbstractArray, y::AbstractVector;
              n_epochs=50, learning_rate=1., print_flag=false,
              compute_accuracy=false, seed=srand(1234), pocket=false,
              shuffle_data=false, average_weights = false)

    #n_features, n_samples = size(X)
    #@assert eltype(X) == eltype(h.W)
    n_features = size(X,1)
    n_samples = size(X,2)

    @assert length(y) == n_samples

    T = eltype(X)
    learning_rate = T(learning_rate)

    y_signal_placeholder = zeros(T, h.n_classes)
    y_preds = zeros(Int64, n_samples)
    data_indices = Array(1:n_samples)
    max_acc = zero(T)

    if pocket
        W_hat = zeros(T, h.n_features, h.n_classes)
        b_hat = zeros(T, h.n_classes)
    end
    if average_weights
        W_his = zeros(T, h.n_features, h.n_classes)
        b_his = zeros(T, h.n_classes)
    end

    #x_views = [view(X, :, m) for m in 1:n_samples]

    for epoch in 1:n_epochs

        n_mistakes = 0
        if shuffle_data
            shuffle!(seed, data_indices)
        end

        @inbounds for m in data_indices
            x = view(X, :, m)
            #x = x_views[m]

            y_hat = predict(h, x, y_signal_placeholder)
            #y_hat = predict(h, x)

            if y[m] != y_hat
                n_mistakes += 1
                ####  wij ← wij − η (yj −tj) · xi

                h.W[:, y[m]]  .= h.W[:, y[m]]  .+ learning_rate .* x
                h.b[y[m]]     .= h.b[y[m]]     .+ learning_rate
                h.W[:, y_hat] .= h.W[:, y_hat] .- learning_rate .* x
                h.b[y_hat]    .= h.b[y_hat]    .- learning_rate
            end
        end

        if compute_accuracy
             @inbounds for m in  data_indices
                 y_preds[m] = predict(h, view(X, :, m), y_signal_placeholder)
            end
            acc = accuracy(y, y_preds)
            push!(h.accuracy, acc)
        else
            acc = (n_samples - n_mistakes)/n_samples
            push!(h.accuracy, acc)
        end

        if pocket
            if acc > max_acc
                max_acc = acc
                copy!(W_hat, h.W)
                copy!(b_hat, h.b)
            end
        end
        if average_weights
            W_his .=  W_his .+ h.W
            b_his .=  b_his .+ h.b
        end

        if print_flag
            print("Epoch: $(epoch) \t Accuracy: $(round(acc,3))\r")
            flush(STDOUT)
        end
    end

    if compute_accuracy && pocket
        h.W .= W_hat
        h.b .= b_hat
    end
    if average_weights
        println("")
        println("history weights\n", W_his[1:5])
        println("weights\n",h.W[1:5])
        println("history weights scaled\n",W_his[1:5]/n_epochs)
        h.W .= W_his/n_epochs
        h.b .= b_his/n_epochs
    end

    return
end


function fit2!(h::MPerceptron, X::AbstractArray, y::AbstractVector;
              n_epochs=50, learning_rate=1., print_flag=false,
              compute_accuracy=false, seed=srand(1234), pocket=false,
              shuffle_data=false, average_weights = false)

    #n_features, n_samples = size(X)
    #@assert eltype(X) == eltype(h.W)
    n_features = size(X,1)
    n_samples = size(X,2)

    @assert length(y) == n_samples

    T = eltype(X)
    learning_rate = T(learning_rate)

    y_signal_placeholder = zeros(T, h.n_classes)
    y_preds = zeros(Int64, n_samples)
    data_indices = Array(1:n_samples)
    max_acc = zero(T)

    if pocket
        W_hat = zeros(T, h.n_features, h.n_classes)
        b_hat = zeros(T, h.n_classes)
    end
    if average_weights
        W_his = zeros(T, h.n_features, h.n_classes)
        b_his = zeros(T, h.n_classes)
    end

    #x_views = [view(X, :, m) for m in 1:n_samples]

    for epoch in 1:n_epochs

        n_mistakes = 0
        if shuffle_data
            shuffle!(seed, data_indices)
        end

        @inbounds for m in data_indices
            x = view(X, :, m)
            y_hat = predict(h, x, y_signal_placeholder)
            y_m = y[m]

            if y[m] != y_hat
                n_mistakes += 1
                for j in 1:n_features
                    h.W[j, y_m]   =  h.W[j, y_m]   .+ learning_rate .* x[j]
                    h.W[j, y_hat] =  h.W[j, y_hat]  .+ learning_rate .* x[j]
                end
                h.b[y_m]      .= h.b[y_m]      .+ learning_rate
                h.b[y_hat]    .= h.b[y_hat]    .- learning_rate
            end
        end

        if compute_accuracy
             @inbounds for m in  data_indices
                 y_preds[m] = predict(h, view(X, :, m), y_signal_placeholder)
            end
            acc = accuracy(y, y_preds)
            push!(h.accuracy, acc)
        else
            acc = (n_samples - n_mistakes)/n_samples
            push!(h.accuracy, acc)
        end

        if pocket
            if acc > max_acc
                max_acc = acc
                copy!(W_hat, h.W)
                copy!(b_hat, h.b)
            end
        end
        if average_weights
            W_his .=  W_his .+ h.W
            b_his .=  b_his .+ h.b
        end

        if print_flag
            print("Epoch: $(epoch) \t Accuracy: $(round(acc,3))\r")
            flush(STDOUT)
        end
    end

    if compute_accuracy && pocket
        h.W .= W_hat
        h.b .= b_hat
    end
    if average_weights
        println("")
        println("history weights\n", W_his[1:5])
        println("weights\n",h.W[1:5])
        println("history weights scaled\n",W_his[1:5]/n_epochs)
        h.W .= W_his/n_epochs
        h.b .= b_his/n_epochs
    end

    return
end

end
