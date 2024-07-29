using JuMP
using SparseArrays
using LinearAlgebra
using Ipopt

################################################
#=
# Test Linear Bilevel
=#
################################################

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

function constraint_lower!(model, x, y; slack = zeros(4))
    @constraints(model, begin
        x +  y + slack[1] <= 8
        4x +  y + slack[2] >= 8
        2x +  y + slack[3] <= 13
        2x - 7y + slack[4] <= 0
    end)
end

function obj_lower!(model, x, y, slack)
    @objective(model, Min, -x + 10 * sum(s^2 for s in slack))
end

function build_lower!(model, x, y, slack)
    obj_lower!(model, x, y, slack)
    constraint_lower!(model, x, y, slack = slack)
end

function memoize(foo::Function)
    last_x, last_f = nothing, nothing
    last_dx, last_dfdx = nothing, nothing
    function foo_i(x::T...) where {T<:Real}
        if T == Float64
            if x !== last_x
                last_x, last_f = x, foo(x...)
            end
            return last_f::T
        else
            if x !== last_dx
                last_dx, last_dfdx = x, foo(x...)
            end
            return last_dfdx::T
        end
    end
    return (x...) -> foo_i(x...)
end

# using BilevelJuMP

# model = BilevelModel(Ipopt.Optimizer, mode = BilevelJuMP.ProductMode(1e-5))

# @variable(Lower(model), x_b)
# @variable(Upper(model), y_b)

# build_upper!(Upper(model), x_b, y_b)
# build_lower!(Lower(model), x_b, y_b, zeros(4))
# optimize!(model)

# objective_value(model) # = 3 * (3.5 * 8/15) + 8/15 # = 6.13...
# value(x_b) # = 3.5 * 8/15 # = 1.86...
# value(y_b) # = 8/15 # = 0.53...

function test_bilevel_linear()
    model_upper = Model(Ipopt.Optimizer)
    model_lower = Model(Ipopt.Optimizer)

    @variable(model_upper, y)
    @variable(model_upper, x_star)
    @variable(model_upper, x_aux)
    @variable(model_lower, x)
    @variable(model_lower, y_p ∈ MOI.Parameter(1.0))
    @variable(model_lower, slack[1:4])

    build_upper!(model_upper, x_star, y)
    constraint_lower!(model_upper, x_aux, y)
    build_lower!(model_lower, x, y_p, slack)
    primal_vars = [x; slack]
    params = [y_p]
    evaluator, cons = create_evaluator(model_lower; x=[primal_vars; params])

    # Define `f(y) = x_star` & `f'(y)`
    function f(y_val)
        set_parameter_value(y_p, y_val)
        optimize!(model_lower)
        @assert is_solved_and_feasible(model_lower)
        return value(x)
    end

    function ∇f0(y_val)
        @assert value(y_p) == y_val
        Δs, sp = compute_sensitivity(evaluator, cons, [0.00]; primal_vars=primal_vars, params=params)
        return Δs[1]
    end

    function ∇f(y_val)
        @assert value(y_p) == y_val
        Δs, sp = compute_sensitivity(evaluator, cons, [0.001]; primal_vars=primal_vars, params=params)
        return Δs[1]
    end

    memoized_f = memoize(f)

    @operator(model_upper, op_f, 1, memoized_f, ∇f)
    @constraint(model_upper, x_star == op_f(y))

    optimize!(model_upper)

    @test objective_value(model_upper) ≈ 6.13 atol=0.05
    @test value(x_star) ≈ 1.86 atol=0.05
    @test value(y) ≈ 0.53 atol=0.05
end

################################################
#=
# Test Non-Linear Bilevel
=#
################################################