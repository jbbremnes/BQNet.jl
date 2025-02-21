#  Customised softplus function that can be applied to all but the first output variable.
#  The function should be used on the output when training with argument increments = true
#
#  
#

using Requires

softplus_bqn(x::Matrix) = vcat(x[1:1, :], softplus(x[2:end, :]))



#  CUDA
try
    
    using CUDA, ChainRulesCore

    function softplus_bqn_kernel!(y, x)
        i, j = threadIdx().x + (blockIdx().x - 1) * blockDim().x, threadIdx().y + (blockIdx().y - 1) * blockDim().y
        if i <= size(x,1) && j <= size(x,2)
            y[i, j] = (i == 1) ? x[i, j] : softplus(x[i, j])
        end
        return nothing
    end
    
    function softplus_bqn(x::CuArray{Float32, 2})
        y = similar(x)
        threads = (16, 16)
        blocks = cld.(size(x), threads)
        @cuda threads=threads blocks=blocks softplus_bqn_kernel!(y, x)
        return y
    end
    
    function ChainRulesCore.rrule(::typeof(softplus_bqn), x::CuArray{Float32, 2})
        y = softplus_bqn(x)  # Compute forward pass output
        
        function softplus_bqn_pullback(Δ)
            grad_x = similar(x)
            threads = (16, 16)
            blocks = cld.(size(x), threads)
            @cuda threads=threads blocks=blocks softplus_bqn_grad_kernel!(grad_x, x, Δ)
            return (NoTangent(), grad_x)
        end
        
        return y, softplus_bqn_pullback
    end
    
    function softplus_bqn_grad_kernel!(grad_x, x, Δ)
        i, j = threadIdx().x + (blockIdx().x - 1) * blockDim().x, threadIdx().y + (blockIdx().y - 1) * blockDim().y
        if i <= size(x,1) && j <= size(x,2)
            if i == 1
                grad_x[i, j] = Δ[i, j]  # Pass-through gradient for first row
            else
                grad_x[i, j] = Δ[i, j] * sigmoid(x[i, j])  # Derivative of softplus
            end
        end
        return nothing
    end

catch e
    
    println("No CUDA")
    
end



##  simple test (move elsewhere)
if false
    using Flux, CUDA, Zygote

    x = cu(randn(Float32, 10, 10))  # Random CuArray input
    y = softplus_bqn(x)             # Forward pass
    grad = Zygote.gradient(x -> sum(softplus_bqn(x)), x)  # Compute gradient



    using Test, CUDA, Zygote
    
    # Test CPU version
    @test softplus_bqn(rand(Float32, 10, 10)) isa Array{Float32, 2}
    
    # Test GPU version (only if CUDA is available)
    if CUDA.functional()
        x_gpu = cu(rand(Float32, 10, 10))
        y_gpu = softplus_bqn(x_gpu)
        @test y_gpu isa CuArray{Float32, 2}
        
        grad = Zygote.gradient(x -> sum(softplus_bqn(x)), x_gpu)
        @test grad isa Tuple  # Should return a valid gradient
    end
    
end

