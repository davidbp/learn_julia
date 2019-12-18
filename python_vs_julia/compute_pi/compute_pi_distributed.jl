using Distributed

@everywhere function get_sample()
    
    x = rand() * 2. - 1.  # uniformly distributed number on x-axis
    y = rand() * 2. - 1.  # uniformly distributed number on y-axis

    r2 = x*x + y*y  # radius squared, in radial coordinates
    if r2 < 1.0
        return 1
    else 
        return 0
    end
end

function compute_pi(N::Int)
    """
    Compute pi with a Monte Carlo simulation of N darts thrown in [-1,1]^2
    Returns estimate of pi
    """
    n_landed_in_circle = @distributed (+) for i = 1:N
        get_sample()
    end

    return n_landed_in_circle / N * 4.0    
end


N= 1000_000_000
t0 = time()
println("Start Computing")
aprox = compute_pi(N)
println("Approximate pi: ", aprox)
println("Total time: ", abs(time()-t0), " seconds")


