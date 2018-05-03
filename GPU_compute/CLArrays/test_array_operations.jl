using CLArrays


sizes = [100,500,1000]


for s in sizes

    srand(123)
    X = rand(Float32,s,s)
    Xcl = CLArray(X)
    aux1 = Xcl * Xcl

    println("result of the sum: ", sum(aux1))
end

