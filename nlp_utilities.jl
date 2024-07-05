using JuMP.MOI
using SparseArrays

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
    MOI.eval_hessian_lagrangian(evaluator, V, value.(x), 1.0, dual.(rows))
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
function find_inequealities(cons::Vector{ConstraintRef})
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
    slack = MOI.get(owner_model(con), CanonicalConstraintFunction(), con)
    return slack
end

function compute_solution_and_bounds(primal_vars, cons)
    num_vars = length(primal_vars)
    ineq_locations = find_inequealities(cons)
    num_ineq = length(ineq_locations)
    slack_vars = [get_slack_inequality(cons[i]) for i in ineq_locations]

    # Primal solution
    X = value.([primal_vars; slack_vars])

    # value and dual of the lower bounds
    V_L = zeros(num_vars+num_ineq)
    X_L = zeros(num_vars+num_ineq)
    for i in 1:num_vars
        if has_lower_bound(primal_vars[i])
            V_L[i] = dual.(LowerBoundRef(primal_vars[i]))
            X_L[i] = JuMP.lower_bound(primal_vars[i])
        end
    end
    for (i, con) in enumerate(cons[ineq_locations])
        V_L[num_vars+i] = dual.(con)
    end
    # value and dual of the upper bounds
    V_U = zeros(num_vars+num_ineq)
    X_U = zeros(num_vars+num_ineq)
    for i in 1:num_vars
        if has_upper_bound(primal_vars[i])
            V_U[i] = dual.(UpperBoundRef(primal_vars[i]))
            X_U[i] = JuMP.upper_bound(primal_vars[i])
        end
    end

    return X, V_L, X_L, V_U, X_U, ineq_locations
end

"""
    compute_derivatives(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}; primal_vars::Vector{VariableRef}=all_primal_vars(model), params::Vector{VariableRef}=all_params(model))

Compute the derivatives of the solution with respect to the parameters without accounting for active set changes.
"""
function compute_derivatives_no_relax(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}, 
    _X::Vector{T}, _V_L::Vector{T}, _X_L::Vector{T}, _V_U::Vector{T}, _X_U::Vector{T}, ineq_locations::Vector{Z}
) where {T<:Real, Z<:Integer}
    @assert all(x -> is_parameter(x), params) "All parameters must be parameters"

    # Setting
    num_vars = length(primal_vars)
    num_parms = length(params)
    num_cons = length(cons)
    num_ineq = length(ineq_locations)
    all_vars = [primal_vars; params]

    # Primal solution
    X = diagm(_X)

    # value and dual of the lower bounds
    V_L = diagm(_V_L)
    X_L = diagm(_X_L)
    # value and dual of the upper bounds
    V_U = diagm(_V_U)
    X_U = diagm(_X_U)

    # Function Derivatives
    hessian, jacobian = compute_optimal_hess_jac(evaluator, cons, all_vars)

    # Hessian of the lagrangian wrt the primal variables
    W = zeros(num_vars + num_ineq, num_vars + num_ineq)
    W[1:num_vars, 1:num_vars] = hessian[1:num_vars, 1:num_vars]
    # Jacobian of the constraints wrt the primal variables
    A = zeros(num_cons, num_vars + num_ineq)
    A[:, 1:num_vars] = jacobian[:, 1:num_vars]
    for (i,j) in enumerate(ineq_locations)
        A[j, num_vars+i] = 1
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
    #     [V_U 0 0 0 (X - X_U)]
    # ]
    len_w = num_vars + num_ineq
    M = zeros(3 * len_w + num_cons, 3 * len_w + num_cons)

    M[1:len_w, 1:len_w] = W
    M[1:len_w, len_w + 1 : len_w + num_cons] = A'
    M[len_w+1:len_w+num_cons, 1:len_w] = A
    M[1:len_w, len_w+num_cons+1:2 * len_w+num_cons] = -I(len_w)
    M[len_w+num_cons+1:2 * len_w+num_cons, 1:len_w] = V_L
    M[len_w+num_cons+1:2 * len_w+num_cons, len_w+num_cons+1:2 * len_w+num_cons] = X - X_L
    M[2 * len_w+num_cons+1:3 * len_w+num_cons, 1:len_w] = V_U
    M[2 * len_w+num_cons+1:3 * len_w+num_cons, 2 * len_w+num_cons+1:3 * len_w+num_cons] = X - X_U
    M[1:len_w, 2 * len_w+num_cons+1:end] = I(len_w)

    # N matrix
    N = [∇ₓₚL ; ∇ₚC; zeros(2 * len_w, num_parms)]

    # Sesitivity of the solution (primal-dual_constraints-dual_bounds) with respect to the parameters
    K = qr(M) # Factorization
    return - (K \ N), K, N
end

function fix_and_relax(E, K, N, r1, ∂p)
    rs = N * ∂p
    # C = −E' inv(K) E
    C = - E' * (K \ E)
    # C ∆ν¯ = E' inv(K) rs − r1
    ∆ν¯ = C \ (E' * (K \ rs) - r1)
    # K ∆s = − (rs + E∆ν¯)
    return K \ (- (rs + E * ∆ν¯))
end

"""
    compute_derivatives(model::Model; primal_vars=all_primal_vars(model), params=all_params(model))

Compute the derivatives of the solution with respect to the parameters.
"""
function compute_derivatives(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}, 
    ∂p::Vector{T}; primal_vars=all_primal_vars(model), params=all_params(model), tol=1e-6
) where {T<:Real}
    num_cons = length(cons)
    # Solution and bounds
    X, V_L, X_L, V_U, X_U, ineq_locations = compute_solution_and_bounds(primal_vars, cons)
    num_w = length(ineq_locations) + length(primal_vars)
    # Compute derivatives
    ∂s, K, N = compute_derivatives_no_relax(evaluator, cons, primal_vars, params, X, V_L, X_L, V_U, X_U, ineq_locations)
    # Linearly appoximated solution
    sp = [X; dual.(cons); V_L; V_U] .+ ∂s * ∂p
    # One-hot vector that signals the bounds that are violated 
    # [X_L<= X <= X_U, dual ∈ R, 0 <= V]
    E = [1.0 * (sp[1:num_w] .> X_U .+ tol) + 1.0 * (sp[1:num_w] .< X_L  .+ tol); zeros(num_cons); sp[num_w+num_cons+1:end] .> 0.0  .+ tol]
    # optimal solution at the violated bounds
    r1 = E .* [X; zeros(num_cons); V_L; V_U]
    if sum(E) > 0
        return fix_and_relax(E, K, N, r1, ∂p), evaluator, cons
    end
    return ∂s
end

function compute_derivatives(model::Model, ∂p::Vector{T}; primal_vars=all_primal_vars(model), params=all_params(model)) where {T<:Real}
    evaluator, cons = create_evaluator(model; x=[primal_vars; params])
    return compute_derivatives(evaluator, cons, ∂p; primal_vars=primal_vars, params=params), evaluator, cons
end