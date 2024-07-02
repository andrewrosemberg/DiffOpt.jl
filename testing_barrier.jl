################################################
# sIPOPT test
################################################

using JuMP
using SparseArrays
using LinearAlgebra
using Ipopt
# using MadNLP
# using KNITRO

include("nlp_utilities.jl")
include("nlp_utilities_test.jl")

############
# Test Case 1
############

# Define the problem
par_model = Model(Ipopt.Optimizer)

# Parameters
@variable(par_model, p ∈ MOI.Parameter(1.0))
@variable(par_model, p2 ∈ MOI.Parameter(2.0))

# Variables
@variable(par_model, x) 
@variable(par_model, y)

# Constraints
@constraint(par_model, con1, y >= p*sin(x)) # NLP Constraint
@constraint(par_model, con2, x + y == p)
@constraint(par_model, con3, p2 * x >= 0.1)
@objective(par_model, Min, (1 - x)^2 + 100 * (y - x^2)^2) # NLP Objective
optimize!(par_model)

# Check local optimality
termination_status(par_model)

############
# Sensitivity Analysis
############

∂s, evaluator, rows = compute_derivatives(par_model)


test_compute_optimal_hess_jacobian()