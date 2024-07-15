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


test_compute_optimal_hess_jacobian()

test_compute_derivatives()

test_compute_derivatives_Finite_Diff()


#######################
# Test Bilevel (WIP)
#######################

using BilevelJuMP, Ipopt

function constraint_upper!(model, x, y)
    @constraints(model, begin
        x <= 5
        y <= 8
        y >= 0
    end)
end

function obj_upper!(model, x, y)
    @objective(model, Min, 3x + y)
end

function build_upper!(model, x, y)
    obj_upper!(model, x, y)
    constraint_upper!(model, x, y)
end

function constraint_lower!(model, x, y)
    @constraints(model, begin
        x +  y <= 8
        4x +  y >= 8
        2x +  y <= 13
        2x - 7y <= 0
    end)
end

function obj_lower!(model, x, y)
    @objective(model, Min, -x)
end

function build_lower!(model, x, y)
    obj_lower!(model, x, y)
    constraint_lower!(model, x, y)
end

### Test BilevelJuMP ###

model = BilevelModel(Ipopt.Optimizer, mode = BilevelJuMP.ProductMode(1e-5))

@variable(Lower(model), x)
@variable(Upper(model), y)

build_upper!(Upper(model), x, y)
build_lower!(Lower(model), x, y)
optimize!(model)

objective_value(model) # = 3 * (3.5 * 8/15) + 8/15 # = 6.13...
value(x) # = 3.5 * 8/15 # = 1.86...
value(y) # = 8/15 # = 0.53...

### Test DiffOpt Non-Convex ###

model_upper = Model(Ipopt.Optimizer)
model_lower = Model(Ipopt.Optimizer)

@variable(model_upper, y)
@variable(model_upper, x_star)
@variable(model_upper, x_aux)
@variable(model_lower, x)
@variable(model_lower, y_p ∈ MOI.Parameter(1.0))

build_upper!(model_upper, x_star, y)
constraint_lower!(model_upper, x_aux, y)
build_lower!(model_lower, x, y_p)

# Define `f(y) = x_star` & `f'(y)`