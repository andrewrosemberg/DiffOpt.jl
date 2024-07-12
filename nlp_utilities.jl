using JuMP
using MathOptInterface
import MathOptInterface: ConstraintSet, CanonicalConstraintFunction
using SparseArrays
using LinearAlgebra

"""
    create_nlp_model(model::JuMP.Model)

Create a Nonlinear Programming (NLP) model from a JuMP model.
"""
function create_nlp_model(model::JuMP.Model)
    rows = Vector{ConstraintRef}(undef, 0)
    nlp = MOI.Nonlinear.Model()
    for (F, S) in list_of_constraint_types(model)
        if F <: VariableRef
            continue  # Skip variable bounds
        end
        for ci in all_constraints(model, F, S)
            push!(rows, ci)
            object = constraint_object(ci)
            MOI.Nonlinear.add_constraint(nlp, object.func, object.set)
        end
    end
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    return nlp, rows
end

"""
    fill_off_diagonal(H)

Filling the off-diagonal elements of a sparse matrix to make it symmetric.
"""
function fill_off_diagonal(H)
    ret = H + H'
    row_vals = SparseArrays.rowvals(ret)
    non_zeros = SparseArrays.nonzeros(ret)
    for col in 1:size(ret, 2)
        for i in SparseArrays.nzrange(ret, col)
            if col == row_vals[i]
                non_zeros[i] /= 2
            end
        end
    end
    return ret
end

"""
    compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Hessian of the Lagrangian.
"""
function compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    MOI.eval_hessian_lagrangian(evaluator, V, value.(x), -1.0, dual.(rows))
    H = SparseArrays.sparse(I, J, V, length(x), length(x))
    return Matrix(fill_off_diagonal(H))
end

"""
    compute_optimal_jacobian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Jacobian of the constraints.
"""
function compute_optimal_jacobian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    jacobian_sparsity = MOI.jacobian_structure(evaluator)
    I = [i for (i, _) in jacobian_sparsity]
    J = [j for (_, j) in jacobian_sparsity]
    V = zeros(length(jacobian_sparsity))
    MOI.eval_constraint_jacobian(evaluator, V, value.(x))
    A = SparseArrays.sparse(I, J, V, length(rows), length(x))
    return Matrix(A)
end

"""
    compute_optimal_hess_jac(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Hessian of the Lagrangian and Jacobian of the constraints.
"""
function compute_optimal_hess_jac(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    hessian = compute_optimal_hessian(evaluator, rows, x)
    jacobian = compute_optimal_jacobian(evaluator, rows, x)
    
    return hessian, jacobian
end

"""
    all_primal_vars(model::Model)

Get all the primal variables in the model.
"""
all_primal_vars(model::Model) = filter(x -> !is_parameter(x), all_variables(model))

"""
    all_params(model::Model)

Get all the parameters in the model.
"""
all_params(model::Model) = filter(x -> is_parameter(x), all_variables(model))

"""
    create_evaluator(model::Model; x=all_variables(model))

Create an evaluator for the model.
"""
function create_evaluator(model::Model; x=all_variables(model))
    nlp, rows = create_nlp_model(model)
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(nlp, backend, index.(x))
    MOI.initialize(evaluator, [:Hess, :Jac])
    return evaluator, rows
end

"""
    is_inequality(con::ConstraintRef)

Check if the constraint is an inequality.
"""
function is_inequality(con::ConstraintRef)
    set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
    return set_type <: MOI.LessThan || set_type <: MOI.GreaterThan
end

"""
    find_inequealities(cons::Vector{ConstraintRef})

Find the indices of the inequality constraints.
"""
function find_inequealities(cons::Vector{C}) where C<:ConstraintRef
    ineq_locations = zeros(length(cons))
    for i in 1:length(cons)
        if is_inequality(cons[i])
            ineq_locations[i] = true
        end
    end
    return findall(x -> x ==1, ineq_locations)
end

"""
    get_slack_inequality(con::ConstraintRef)

Get the reference to the canonical function that is equivalent to the slack variable of the inequality constraint.
"""
function get_slack_inequality(con::ConstraintRef)
    set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
    obj = constraint_object(con)
    if set_type <: MOI.LessThan
        # c(x) <= b --> slack = -c(x) + b | slack >= 0
        return - obj.func + obj.set.upper 
    end
    return obj.func - obj.set.lower
end

function compute_solution_and_bounds(primal_vars, cons)
    num_vars = length(primal_vars)
    ineq_locations = find_inequealities(cons)
    num_ineq = length(ineq_locations)
    slack_vars = [get_slack_inequality(cons[i]) for i in ineq_locations]
    has_up =  findall(x -> has_upper_bound(x), primal_vars)
    has_low = findall(x -> has_lower_bound(x), primal_vars)

    # Primal solution
    X = value.([primal_vars; slack_vars])

    # value and dual of the lower bounds
    V_L = zeros(num_vars+num_ineq)
    X_L = zeros(num_vars+num_ineq)
    for (i, j) in enumerate(has_low)
        V_L[i] = dual.(LowerBoundRef(primal_vars[j]))
        X_L[i] = JuMP.lower_bound(primal_vars[j])
    end
    for (i, con) in enumerate(cons[ineq_locations])
        V_L[num_vars+i] = dual.(con)
    end
    # value and dual of the upper bounds
    V_U = zeros(num_vars+num_ineq)
    X_U = zeros(num_vars+num_ineq)
    for (i, j) in enumerate(has_up)
        V_U[i] = dual.(UpperBoundRef(primal_vars[j]))
        X_U[i] = JuMP.upper_bound(primal_vars[j])
    end

    return X, V_L, X_L, V_U, X_U, ineq_locations, has_up, vcat(has_low, collect(num_vars+1:num_vars+num_ineq))
end

function build_M_N(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}, 
    _X::Vector{T}, _V_L::Vector{T}, _X_L::Vector{T}, _V_U::Vector{T}, _X_U::Vector{T}, ineq_locations::Vector{Z},
    has_up::Vector{Z}, has_low::Vector{Z}
) where {T<:Real, Z<:Integer}
    @assert all(x -> is_parameter(x), params) "All parameters must be parameters"

    # Setting
    num_vars = length(primal_vars)
    num_parms = length(params)
    num_cons = length(cons)
    num_ineq = length(ineq_locations)
    all_vars = [primal_vars; params]
    num_low = length(has_low)
    num_up = length(has_up)

    # Primal solution
    X_lb = zeros(num_low, num_low)
    X_ub = zeros(num_up, num_up)
    V_L = zeros(num_low, num_vars + num_ineq)
    V_U = zeros(num_up, num_vars + num_ineq)
    I_L = zeros(num_vars + num_ineq,  num_low)
    I_U = zeros(num_vars + num_ineq,  num_up)

    # value and dual of the lower bounds
    for (i, j) in enumerate(has_low)
        V_L[i, j] = _V_L[j]
        X_lb[i, i] = _X[j] - _X_L[j]
        I_L[j, i] = -1
    end
    # value and dual of the upper bounds
    for (i, j) in enumerate(has_up)
        V_U[i, j] = _V_U[j]
        X_ub[i, i] = _X_U[j] - _X[j]
        I_U[j, i] = 1
    end

    # Function Derivatives
    hessian, jacobian = compute_optimal_hess_jac(evaluator, cons, all_vars)

    # Hessian of the lagrangian wrt the primal variables
    W = zeros(num_vars + num_ineq, num_vars + num_ineq)
    W[1:num_vars, 1:num_vars] = hessian[1:num_vars, 1:num_vars]
    # Jacobian of the constraints wrt the primal variables
    A = zeros(num_cons, num_vars + num_ineq)
    A[:, 1:num_vars] = jacobian[:, 1:num_vars]
    for (i,j) in enumerate(ineq_locations)
        A[j, num_vars+i] = -1
    end
    # Partial second derivative of the lagrangian wrt primal solution and parameters
    ∇ₓₚL = zeros(num_vars + num_ineq, num_parms)
    ∇ₓₚL[1:num_vars, :] = hessian[1:num_vars, num_vars+1:end]
    # Partial derivative of the equality constraintswith wrt parameters
    ∇ₚC = jacobian[:, num_vars+1:end]

    # M matrix
    # M = [
    #     [W A' -I I];
    #     [A 0 0 0];
    #     [V_L 0 (X - X_L) 0]
    #     [V_U 0 0 0 (X_U - X)]
    # ]
    len_w = num_vars + num_ineq
    M = zeros(len_w + num_cons + num_low + num_up, len_w + num_cons + num_low + num_up)

    M[1:len_w, 1:len_w] = W
    M[1:len_w, len_w + 1 : len_w + num_cons] = A'
    M[len_w+1:len_w+num_cons, 1:len_w] = A
    M[1:len_w, len_w+num_cons+1:len_w+num_cons+num_low] = I_L
    M[len_w+num_cons+1:len_w+num_cons+num_low, 1:len_w] = V_L
    M[len_w+num_cons+1:len_w+num_cons+num_low, len_w+num_cons+1:len_w+num_cons+num_low] = X_lb
    M[len_w+num_cons+num_low+1:len_w+num_cons+num_low+num_up, 1:len_w] = V_U
    M[len_w+num_cons+num_low+1:len_w+num_cons+num_low+num_up, len_w+num_cons+num_low+1:len_w+num_cons+num_low+num_up] = X_ub
    M[1:len_w, len_w+num_cons+num_low+1:end] = I_U

    # N matrix
    N = [∇ₓₚL ; ∇ₚC; zeros(num_low + num_up, num_parms)]

    return M, N
end

"""
    compute_derivatives(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}; primal_vars::Vector{VariableRef}=all_primal_vars(model), params::Vector{VariableRef}=all_params(model))

Compute the derivatives of the solution with respect to the parameters without accounting for active set changes.
"""
function compute_derivatives_no_relax(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}, 
    _X::Vector{T}, _V_L::Vector{T}, _X_L::Vector{T}, _V_U::Vector{T}, _X_U::Vector{T}, ineq_locations::Vector{Z},
    has_up::Vector{Z}, has_low::Vector{Z}
) where {T<:Real, Z<:Integer}
    num_bounds = length(has_up) + length(has_low)
    M, N = build_M_N(evaluator, cons, primal_vars, params, _X, _V_L, _X_L, _V_U, _X_U, ineq_locations, has_up, has_low)

    # Sesitivity of the solution (primal-dual_constraints-dual_bounds) with respect to the parameters
    K = qr(M) # Factorization
    ∂s = - (K \ N) # Sensitivity
    ∂s[end-num_bounds+1:end,:] = ∂s[end-num_bounds+1:end,:]  .* -1.0 # Correcting the sign of the bounds duals for the standard form
    return ∂s, K, N
end

function fix_and_relax(E, K, N, r1, Δp, num_bounds)
    rs = N * Δp
    # C = −E' inv(K) E
    C = - E' * (K \ E)
    # C ∆ν¯ = E' inv(K) rs − r1
    ∆ν¯ = C \ (E' * (K \ rs) - r1)
    # K ∆s = − (rs + E∆ν¯)
    ∆s = K \ (- (rs + E * ∆ν¯))
    ∆s[end-num_bounds+1:end] = ∆s[end-num_bounds+1:end] .* -1.0 # Correcting the sign of the bounds duals for the standard form
    return ∆s
end

function approximate_solution(X, Λ, V_L, V_U, Δs)
    return [X; Λ; V_L; V_U] .+ Δs
end

function find_violations(X, sp, X_L, X_U, V_U, V_L, has_up, has_low, num_cons, tol)
    num_w = length(X)
    num_low = length(has_low)
    num_up = length(has_up)
    
    _E = []
    r1 = []
    for (j, i) in enumerate(has_low)
        if sp[i] < X_L[i] - tol
            push!(_E, i)
            push!(r1, X[i] - X_L[i])
        end
        if sp[num_w+num_cons+j] < -tol
            push!(_E, num_w+num_cons+j)
            push!(r1, V_L[i])
        end
    end
    for (j, i) in enumerate(has_up)
        if sp[i] > X_U[i] + tol
            push!(_E, i)
            push!(r1, X_U[i] - X[i])
        end
        if sp[num_w+num_cons+num_low+j] < -tol
            push!(_E, num_w+num_cons+num_low+j)
            push!(r1, V_U[i])
        end
    end
    
    E = zeros(num_w + num_cons + num_low + num_up, length(_E))
    for (i, j) in enumerate(_E)
        E[j, i] = 1
    end

    return E, r1
end

"""
    compute_derivatives(model::Model; primal_vars=all_primal_vars(model), params=all_params(model))

Compute the derivatives of the solution with respect to the parameters.
"""
function compute_derivatives(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}, 
    Δp::Vector{T}; primal_vars=all_primal_vars(model), params=all_params(model), tol=1e-8
) where {T<:Real}
    num_cons = length(cons)
    # Solution and bounds
    X, V_L, X_L, V_U, X_U, ineq_locations, has_up, has_low = compute_solution_and_bounds(primal_vars, cons)
    # Compute derivatives
    # ∂s = [∂x; ∂λ; ∂ν_L; ∂ν_U]
    ∂s, K, N = compute_derivatives_no_relax(evaluator, cons, primal_vars, params, X, V_L, X_L, V_U, X_U, ineq_locations, has_up, has_low)
    Δs = ∂s * Δp
    Λ = -dual.(cons)
    sp = approximate_solution(X, Λ, V_L[has_low], V_U[has_up], Δs)
    # Linearly appoximated solution
    E, r1 = find_violations(X, sp, X_L, X_U, V_U, V_L, has_up, has_low, num_cons, tol)
    if !isempty(r1)
        @warn "Relaxation needed"
        num_bounds = length(has_up) + length(has_low)
        Δs = fix_and_relax(E, K, N, r1, Δp, num_bounds)
        sp = approximate_solution(X, Λ, V_L[has_low], V_U[has_up], Δs)
    end
    return Δs, sp
end

function compute_derivatives(model::Model, Δp::Vector{T}; primal_vars=all_primal_vars(model), params=all_params(model)) where {T<:Real}
    evaluator, cons = create_evaluator(model; x=[primal_vars; params])
    return compute_derivatives(evaluator, cons, Δp; primal_vars=primal_vars, params=params), evaluator, cons
end
