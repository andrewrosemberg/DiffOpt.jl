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

############
# Test Case
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
# Retrieve important quantities
############

# Primal variables
primal_vars = [x; y]
num_vars = length(primal_vars)
params = [p; p2]
num_parms = length(params)
all_vars = [primal_vars; params]

hessian, jacobian, nlp, cons = compute_optimal_hess_jac(par_model; x=all_vars)

W = hessian[1:num_vars, 1:num_vars]
A = jacobian[:, 1:num_vars]

# ∇ₓₚL (Partial second derivative of the lagrangian wrt primal solution and parameters)
∇ₓₚL = hessian[num_vars+1:end, 1:num_vars]
# ∇ₚC (partial derivative of the equality constraintswith wrt parameters).
∇ₚC = jacobian[:, num_vars+1:end]

############
# Calculate Sensitivity
############
num_cons = length(cons)
X = diagm(value.(primal_vars))

# dual of the bounds
bound_duals = zeros(length(primal_vars))
for i in 1:length(primal_vars)
    if has_lower_bound(primal_vars[i])
        bound_duals[i] = dual.(LowerBoundRef(primal_vars[i]))
    end
    if has_upper_bound(primal_vars[i])
        bound_duals[i] -= dual.(UpperBoundRef(primal_vars[i]))
    end
end
V = diagm(bound_duals)

# M matrix
M = zeros(num_vars * 2 + num_cons, num_vars * 2 + num_cons)

# M = [
#     [W A' -I];
#     [A 0 0];
#     [V 0 X]
# ]

M[1:num_vars, 1:num_vars] = W
M[1:num_vars, num_vars+1:num_vars+num_cons] = A'
M[num_vars+1:num_vars+num_cons, 1:num_vars] = A
M[1:num_vars, num_vars+num_cons+1:end] = -I(num_vars)
M[num_vars+num_cons+1:end, 1:num_vars] = V
M[num_vars+num_cons+1:end, num_vars+num_cons+1:end] = X

N = [∇ₓₚL ; ∇ₚC; zeros(num_vars, num_parms)]

# sesitivity of the solution (primal-dual_constraints-dual_bounds) with respect to the parameters
∂s = pinv(M) * N