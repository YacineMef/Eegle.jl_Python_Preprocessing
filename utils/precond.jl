#= 
 Copyright Marco Congedo, PhD, CNRS, GIPSA-lab, Grenoble, France

 Apply the (pre)conditioning in 
 https://marco-congedo.github.io/PosDefManifoldML.jl/stable/conditioners/
 to tensors of dimension (k x n x n).

 Refer to this page for designing pre-conditioning pipelines.
 The pipeline given in the example usage suits MI BCI data.

 Two methods `pre_cond` are declared, 
 one taking as input training ad test data, and
 another taking as input training, test and validation data.

 The first time you execute this unit, run on julia's REPL:
 ]add PosDefManifold, PosDefManifoldML

=#

module Precond

using PosDefManifold, PosDefManifoldML, LinearAlgebra

#################################################################
## Utilities

# Function to determine the dimensionality reduction given the number of electrodes `n`
# if `n`≤ 64, return n, otherwise return an integer smaller than `n`.
function set_eVar(n::Int)
    n≤64 && return n
    return round(Int, 64+32*(pi/ℯ)*log2((n/64)))
end


# Array of k nxn Hermitian matrices (ℍVector) to tensor (n, n, k)
function ℍVector_to_tensor(Xs::ℍVector, T::Type=eltype(Xs[1])) 
    n = size(Xs[1], 1)
    B = length(Xs)
    Xt = Array{T, 3}(undef, B, n, n)
    for i in 1:B
        Xt[i, :, :] = T.(Matrix(Xs[i]))
    end
    return Xt
end

◻ = ℍVector_to_tensor # alias

# Tensor (n, n, k) to ℍVector nxn Hermitian matrices
function tensor_to_ℍVector(Xt::Array{T, 3}; type::Type=Float64) where {T<:Real}
    ℍVector([Hermitian(type.(Xt[i, :, :])) for i in 1:size(Xt, 1)])
end

#################################################################


# General function taking as input a train and test tensor and a pre-coditioning pipeline 
# and returning the pre-conditioned tensors.
# C_test should include the testing and validation data. See next method for separate
# testing and validation data.
# First, we convert the input data in vector of matrices,
# then we fit the pipeline on training data and apply it to the training data as well.
# Finally, we apply it on testing data and convert the training and testing data into tensors
function pre_cond(C_train, C_test, pipeline::Pipeline) 
    C_train = Array(C_train)
    C_test  = Array(C_test)
    C_trainAsArray = tensor_to_ℍVector(C_train)
    pipe = fit!(C_trainAsArray, pipeline);
 
    # For the test (and validation) data we apply the fitted pipeline
    C_testAsArray = tensor_to_ℍVector(C_test)
    transform!(C_testAsArray, pipe);  

    return ◻(C_trainAsArray), ◻(C_testAsArray) 
end

# As the previous method, but now take as input the tensors for training, test and validation
# and return the pre-conditioned tensors
function pre_cond(C_train, C_test, C_val, pipeline::Pipeline)
    C_train = Array(C_train)
    C_test  = Array(C_test)
    C_val   = Array(C_val)

    C_trainAsArray = tensor_to_ℍVector(C_train)
    pipe = fit!(C_trainAsArray, pipeline);
 
    # For the test data we apply the fitted pipeline
    C_testAsArray = tensor_to_ℍVector(C_test)
    transform!(C_testAsArray, pipe);  

    # For the validation data we apply the fitted pipeline
    C_valAsArray = tensor_to_ℍVector(C_val)
    transform!(C_valAsArray, pipe);  

    return ◻(C_trainAsArray), ◻(C_testAsArray), ◻(C_valAsArray)  
end


###############################################################################
# # Example usage

# # Random data as an example
# n = 2
# k_train = 5
# k_test = 3
# C_train = ◻(randP(n, k_train))
# C_test = ◻(randP(n, k_test))

# # Define the pre-conditioning pipeline for MI
# # NB, we eVar = n we force the recentering NOT to reduce dimension.
# # For allowing a parsimonious representation of the data, set eVar = v,
# # where v is the explained variance parameter introduced in the article at page 4.
# # For MI a good value is v=0.9999, which basically retain all the variance on the data,
# # thus no information whatsoever is lost. Still, the dimension will be reduced
# # for data with a lot of electrodes.
# # Note, that in this case different sessions will likely be reduced 
# # to different dimensions, however the training and testing data of the same session 
# # will always have the same dimension.
# metric = PosDefManifold.Euclidean
# pipeline = @→ Tikhonov(1e-8) → Recenter(metric; eVar=n) → Equalize();

# # just call
# C_train_new, C_test_new = pre_cond(C_train, C_test, pipeline)
###############################################################################



###########################################################
# Test the unit:
# run `test_pre_cond()` in the REPL, you should see a peach
function test_pre_cond()
    n = 10
    k_train = 50
    k_test = 10
    P_train = randP(n, k_train)
    C_train = ◻(P_train)
    P_test = randP(n, k_test)
    C_test = ◻(P_test)

    # Define the pre-conditioning pipeline for MI
    metric = PosDefManifold.Euclidean
    pipeline = @→ Tikhonov(1e-8) → Recenter(metric; eVar=n) → Equalize();

    pipe = fit!(P_train, pipeline);
    transform!(P_test, pipe); 

    C_train_new, C_test_new = pre_cond(C_train, C_test, pipeline)

    P_train_new = tensor_to_ℍVector(C_train_new)
    P_test_new = tensor_to_ℍVector(C_test_new)

    return sum(P_train .== P_train_new) == length(P_train) &&
        sum(P_test .== P_test_new) == length(P_test) ? "🍑" : "⚠"
end
###########################################################

# helper to create a pipeline
function make_pipeline(alpha :: Float64, precond_explVar, n::Int)
    metric = PosDefManifold.Euclidean
    return @→ Tikhonov(alpha) → Recenter(metric; eVar = precond_explVar == 0 ? set_eVar(n) : precond_explVar) → Equalize()   # essayer les 2 : avec et sans equalize 
end

export pre_cond, test_pre_cond, make_pipeline
end