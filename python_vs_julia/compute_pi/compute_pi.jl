function compute_pi(N::Int)
    """
    Compute pi with a Monte Carlo simulation of N darts thrown in [-1,1]^2
    Returns estimate of pi
    """
    n_landed_in_circle = 0  # counts number of points that have radial coordinate < 1, i.e. in circle
    for i = 1:N
        x = rand() * 2. - 1.  # uniformly distributed number on x-axis
        y = rand() * 2. - 1.  # uniformly distributed number on y-axis

        r2 = x*x + y*y  # radius squared, in radial coordinates
        if r2 < 1.0
            n_landed_in_circle += 1
        end
    end

    return n_landed_in_circle / N * 4.0    
end

N= 100_000_000
t0 = time()
println("Start Computing")
aprox = compute_pi(N)
println("Approximate pi: ", aprox)
println("Total time: ", abs(time()-t0), " seconds")

