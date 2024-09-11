using JuMP
using MathOptInterface
import MathOptInterface: ConstraintSet, CanonicalConstraintFunction
using SparseArrays
using LinearAlgebra
import JuMP.index

"""
    create_nlp_model(model::JuMP.Model)

Create a Nonlinear Programming (NLP) model from a JuMP model.
"""
function create_nlp_model(model::JuMP.Model)
    rows = Vector{ConstraintRef}(undef, 0)
    nlp = MOI.Nonlinear.Model()
    for (F, S) in list_of_constraint_types(model)
        if F <: VariableRef && !(S <: MathOptInterface.EqualTo{Float64})
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

sense_mult(x) = JuMP.objective_sense(owner_model(x)) == MOI.MIN_SENSE ? 1.0 : -1.0
sense_mult(x::Vector) = sense_mult(x[1])

"""
    compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Hessian of the Lagrangian.
"""
function compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    sense_multiplier = objective_sense(owner_model(x[1])) == MOI.MIN_SENSE ? 1.0 : -1.0
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    # The signals are being sdjusted to match the Ipopt convention (inner.mult_g)
    # but we don't know if we need to adjust the objective function multiplier
    MOI.eval_hessian_lagrangian(evaluator, V, value.(x), 1.0, - sense_multiplier * dual.(rows))
    H = SparseArrays.sparse(I, J, V, length(x), length(x))
    return fill_off_diagonal(H)
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
    return A
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
JuMP.index(x::JuMP.Containers.DenseAxisArray) = index.(x).data

function create_evaluator(model::Model; x=all_variables(model))
    nlp, rows = create_nlp_model(model)
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(nlp, backend, vcat(index.(x)...))
    MOI.initialize(evaluator, [:Hess, :Jac])
    return evaluator, rows
end

"""
    is_inequality(con::ConstraintRef)

Check if the constraint is an inequality.
"""
# function is_inequality(con::ConstraintRef)
#     set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
#     return set_type <: MOI.LessThan || set_type <: MOI.GreaterThan
# end

function is_less_inequality(con::ConstraintRef)
    set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
    return set_type <: MOI.LessThan
end

function is_greater_inequality(con::ConstraintRef)
    set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
    return set_type <: MOI.GreaterThan
end

function is_inequality(con::ConstraintRef)
    return is_less_inequality(con) || is_greater_inequality(con)
end

"""
    find_inequealities(cons::Vector{ConstraintRef})

Find the indices of the inequality constraints.
"""
function find_inequealities(cons::Vector{C}) where C<:ConstraintRef
    leq_locations = zeros(length(cons))
    geq_locations = zeros(length(cons))
    for i in 1:length(cons)
        if is_less_inequality(cons[i])
            leq_locations[i] = true
        end
        if is_greater_inequality(cons[i])
            geq_locations[i] = true
        end
    end
    return findall(x -> x ==1, leq_locations), findall(x -> x ==1, geq_locations)
end

"""
    get_slack_inequality(con::ConstraintRef)

Get the reference to the canonical function that is equivalent to the slack variable of the inequality constraint.
"""
function get_slack_inequality(con::ConstraintRef)
    set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
    obj = constraint_object(con)
    if set_type <: MOI.LessThan
        # c(x) <= b --> slack = c(x) - b | slack <= 0
        return obj.func - obj.set.upper 
    end
    # c(x) >= b --> slack = c(x) - b | slack >= 0
    return obj.func - obj.set.lower
end

"""
    compute_solution_and_bounds(primal_vars::Vector{VariableRef}, cons::Vector{C}) where C<:ConstraintRef

Compute the solution and bounds of the primal variables.
"""
function compute_solution_and_bounds(primal_vars::Vector{VariableRef}, cons::Vector{C}) where {C<:ConstraintRef}
    sense_multiplier = sense_mult(primal_vars)
    num_vars = length(primal_vars)
    leq_locations, geq_locations = find_inequealities(cons)
    ineq_locations = vcat(geq_locations, leq_locations)
    num_leq = length(leq_locations)
    num_geq = length(geq_locations)
    num_ineq = num_leq + num_geq
    slack_vars = [get_slack_inequality(cons[i]) for i in ineq_locations]
    has_up =  findall(x -> has_upper_bound(x), primal_vars)
    has_low = findall(x -> has_lower_bound(x), primal_vars)

    # Primal solution
    X = value.([primal_vars; slack_vars])

    # value and dual of the lower bounds
    V_L = spzeros(num_vars+num_ineq)
    X_L = spzeros(num_vars+num_ineq)
    for (i, j) in enumerate(has_low)
        V_L[i] = dual.(LowerBoundRef(primal_vars[j])) * sense_multiplier
        #
        if sense_multiplier == 1.0
            V_L[i] <= -1e-6 && @info "Dual of lower bound must be positive" i V_L[i]
        else
            V_L[i] >= 1e-6 && @info "Dual of lower bound must be negative" i V_L[i]
        end
        #
        X_L[i] = JuMP.lower_bound(primal_vars[j])
    end
    for (i, con) in enumerate(cons[geq_locations])
        # By convention jump dual will allways be positive for geq constraints
        # but for ipopt it will be positive if min problem and negative if max problem
        V_L[num_vars+i] = dual.(con) * (sense_multiplier)
        #
        if sense_multiplier == 1.0
            V_L[num_vars+i] <= -1e-6 && @info "Dual of geq constraint must be positive" i V_L[num_vars+i]
        else
            V_L[num_vars+i] >= 1e-6 && @info "Dual of geq constraint must be negative" i V_L[num_vars+i]
        end
    end
    # value and dual of the upper bounds
    V_U = spzeros(num_vars+num_ineq)
    X_U = spzeros(num_vars+num_ineq)
    for (i, j) in enumerate(has_up)
        V_U[i] = dual.(UpperBoundRef(primal_vars[j])) * (- sense_multiplier)
        #
        if sense_multiplier == 1.0
            V_U[i] <= -1e-6 && @info "Dual of upper bound must be positive" i V_U[i]
        else
            V_U[i] >= 1e-6 && @info "Dual of upper bound must be negative" i V_U[i]
        end
        #
        X_U[i] = JuMP.upper_bound(primal_vars[j])
    end
    for (i, con) in enumerate(cons[leq_locations])
        # By convention jump dual will allways be negative for leq constraints
        # but for ipopt it will be positive if min problem and negative if max problem
        V_U[num_vars+i] = dual.(con) * (- sense_multiplier)
        #
        if sense_multiplier == 1.0
            V_U[num_vars+i] <= -1e-6 && @info "Dual of leq constraint must be positive" i V_U[num_vars+i]
        else
            V_U[num_vars+i] >= 1e-6 && @info "Dual of leq constraint must be negative" i V_U[num_vars+i]
        end
    end
    return X, V_L, X_L, V_U, X_U, leq_locations, geq_locations, ineq_locations, vcat(has_up, collect(num_vars+num_geq+1:num_vars+num_geq+num_leq)), vcat(has_low, collect(num_vars+1:num_vars+num_geq))
end

"""
    build_M_N(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}, 
    _X::Vector, _V_L::Vector, _X_L::Vector, _V_U::Vector, _X_U::Vector, ineq_locations::Vector{Z},
    has_up::Vector{Z}, has_low::Vector{Z}
) where {Z<:Integer}

Build the M (KKT Jacobian w.r.t. solution) and N (KKT Jacobian w.r.t. parameters) matrices for the sensitivity analysis.
"""
function build_M_N(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}, 
    _X::AbstractVector, _V_L::AbstractVector, _X_L::AbstractVector, _V_U::AbstractVector, _X_U::AbstractVector, leq_locations::Vector{Z}, geq_locations::Vector{Z}, ineq_locations::Vector{Z},
    has_up::Vector{Z}, has_low::Vector{Z}
) where {Z<:Integer}
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
    X_lb = spzeros(num_low, num_low)
    X_ub = spzeros(num_up, num_up)
    V_L = spzeros(num_low, num_vars + num_ineq)
    V_U = spzeros(num_up, num_vars + num_ineq)
    I_L = spzeros(num_vars + num_ineq,  num_low)
    I_U = spzeros(num_vars + num_ineq,  num_up)

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
    W = spzeros(num_vars + num_ineq, num_vars + num_ineq)
    W[1:num_vars, 1:num_vars] = hessian[1:num_vars, 1:num_vars]
    # Jacobian of the constraints
    A = spzeros(num_cons, num_vars + num_ineq)
    # A is the Jacobian of: c(x) = b and c(x) <= b and c(x) >= b, possibly all mixed up.
    # Each of the will be re-written as:
    # c(x) - b = 0
    # c(x) - b - su = 0, su <= 0
    # c(x) - b - sl = 0, sl >= 0
    # Jacobian of the constraints wrt the primal variables
    A[:, 1:num_vars] = jacobian[:, 1:num_vars]
    # Jacobian of the constraints wrt the slack variables
    for (i,j) in enumerate(geq_locations)
        A[j, num_vars+i] = -1
    end
    for (i,j) in enumerate(leq_locations)
        A[j, num_vars+i] = -1
    end
    # Partial second derivative of the lagrangian wrt primal solution and parameters
    ∇ₓₚL = spzeros(num_vars + num_ineq, num_parms)
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
    M = spzeros(len_w + num_cons + num_low + num_up, len_w + num_cons + num_low + num_up)

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
    # N = [∇ₓₚL ; ∇ₚC; zeros(num_low + num_up, num_parms)]
    N = spzeros(len_w + num_cons + num_low + num_up, num_parms)
    N[1:len_w, :] = ∇ₓₚL
    N[len_w+1:len_w+num_cons, :] = ∇ₚC

    return M, N
end

function inertia_corrector_factorization(M::SparseMatrixCSC, num_w, num_cons; st=1e-6, max_corrections=50)
    # Factorization
    K = lu(M; check=false)
    # Inertia correction
    status = K.status
    num_c = 0
    diag_mat = ones(size(M, 1))
    diag_mat[num_w+1:num_w+num_cons] .= -1
    diag_mat = sparse(diagm(diag_mat))
    while status == 1 && num_c < max_corrections
        println("Inertia correction")
        M = M + st * diag_mat
        K = lu(M; check=false)
        status = K.status
        num_c += 1
    end
    if status != 0
        @warn "Inertia correction failed"
        return nothing
    end
    return K
end

function inertia_corrector_factorization(M; st=1e-6, max_corrections=50)
    num_c = 0
    if cond(M) > 1/st
        @warn "Inertia correction"
        M = M + st * I(size(M, 1))
        num_c += 1
    end
    while cond(M) > 1/st && num_c < max_corrections
        M = M + st * I(size(M, 1))
        num_c += 1
    end
    if num_c == max_corrections
        @warn "Inertia correction failed"
        return nothing
    end
    return lu(M)
end

"""
    compute_derivatives_no_relax(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
        primal_vars::Vector{VariableRef}, params::Vector{VariableRef},
        _X::Vector, _V_L::Vector, _X_L::Vector, _V_U::Vector, _X_U::Vector, ineq_locations::Vector{Z},
        has_up::Vector{Z}, has_low::Vector{Z}
    ) where {Z<:Integer}

Compute the derivatives of the solution w.r.t. the parameters without accounting for active set changes.
"""
function compute_derivatives_no_relax(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef},
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}, 
    _X::AbstractVector, _V_L::AbstractVector, _X_L::AbstractVector, _V_U::AbstractVector, _X_U::AbstractVector, leq_locations::Vector{Z}, geq_locations::Vector{Z}, ineq_locations::Vector{Z},
    has_up::Vector{Z}, has_low::Vector{Z}
) where {Z<:Integer}
    num_bounds = length(has_up) + length(has_low)
    M, N = build_M_N(evaluator, cons, primal_vars, params, _X, _V_L, _X_L, _V_U, _X_U, leq_locations, geq_locations, ineq_locations, has_up, has_low)

    # Sesitivity of the solution (primal-dual_constraints-dual_bounds) w.r.t. the parameters
    K = inertia_corrector_factorization(M, length(primal_vars) + length(ineq_locations), length(cons)) # Factorization
    if isnothing(K)
        return zeros(size(M, 1), size(N, 2)), K, N
    end
    ∂s = zeros(size(M, 1), size(N, 2))
    # ∂s = - (K \ N) # Sensitivity
    ldiv!(∂s, K, N)
    ∂s = - ∂s

    ## Adjust signs based on JuMP convention
    sense_multiplier = sense_mult(primal_vars)
    num_lower = length(has_low)
    num_w = length(_X)
    num_cons = length(cons)
    # Duals
    ∂s[num_w+1:num_w+num_cons, :] *= -sense_multiplier
    # Dual bounds lower
    ∂s[num_w+num_cons+1:num_w+num_cons+num_lower, :] *= sense_multiplier
    # Dual bounds upper
    ∂s[num_w+num_cons+num_lower+1:end, :] *= -sense_multiplier
    
    return ∂s, K, N
end

"""
    fix_and_relax(E, K, N, r1, Δp, num_w, num_cons)

Fix the violations and relax complementary slackness.
"""
function fix_and_relax(E, K, N, r1, Δp, num_w, num_cons, num_lower, sense_multiplier)
    rs = (N * Δp)[:,:]
    # C = −E' inv(K) E
    # C = - E' * (K \ E)
    C = zeros(size(K, 1), size(E, 2))
    ldiv!(C, K, E)
    C = - E' * C
    # C ∆ν¯ = E' inv(K) rs − r1
    # ∆ν¯ = C \ (E' * (K \ rs) - r1)
    aux = zeros(size(K, 1), size(rs, 2))
    ldiv!(aux, K, rs)
    C_fact = inertia_corrector_factorization(C)
    if isnothing(C_fact)
        return zeros(size(K, 1), size(aux, 2))
    end
    ∆ν¯ = C_fact \ (E' * aux - r1)
    # K ∆s = − (rs + E∆ν¯)
    # ∆s = K \ (- (rs + E * ∆ν¯))
    aux = - (rs + E * ∆ν¯)[:,:]
    ∆s = zeros(size(K, 1), size(aux, 2))
    ldiv!(∆s, K, aux)

    ## Adjust signs based on JuMP convention
    # Duals
    ∆s[num_w+1:num_w+num_cons, :] *= -sense_multiplier
    # Dual bounds lower
    ∆s[num_w+num_cons+1:num_w+num_cons+num_lower, :] *= sense_multiplier
    # Dual bounds upper
    ∆s[num_w+num_cons+num_lower+1:end, :] *= -sense_multiplier

    return ∆s
end

"""
    approximate_solution(X, Λ, V_L, V_U, Δs)

First order approximation the primal-dual solution.
"""
function approximate_solution(X::Vector{T}, Λ, V_L, V_U, Δs) where T
    num_var = length(X)
    num_cons = length(Λ)
    num_low = length(V_L)
    num_up = length(V_U)
    sp = Vector{T}(undef, num_var + num_cons + num_low + num_up)
    sp[1:num_var] = X
    sp[num_var+1:num_var+num_cons] = Λ
    sp[num_var+num_cons+1:num_var+num_cons+num_low] = V_L
    sp[num_var+num_cons+num_low+1:num_var+num_cons+num_low+num_up] = V_U
    sp = sp + Δs

    return sp
end

"""
    find_violations(X, sp, X_L, X_U, V_U, V_L, has_up, has_low, num_cons, tol)

Find the bound violations of the primal-dual solution first order approximation based on the (unrelaxed) sensitivity estimation.
"""
function find_violations(X, sp, X_L, X_U, V_U, V_L, has_up, has_low, num_cons, tol) #, ismin)
    num_w = length(X)
    num_low = length(has_low)
    num_up = length(has_up)
    
    _E = []
    r1 = []
    for (j, i) in enumerate(has_low)
        if sp[i] < X_L[i] - tol
            println("Violation LB Var: ", i, " ", sp[i], " ", X_L[i])
            push!(_E, i)
            push!(r1, X[i] - X_L[i])
        end
        # if ismin
            if sp[num_w+num_cons+j] < -tol
                println("Violation LB Dual: ", i, " ", sp[num_w+num_cons+j], " ", V_L[i])
                push!(_E, num_w+num_cons+j)
                push!(r1, V_L[i])
            end
        # else
        #     if sp[num_w+num_cons+j] > tol
        #         println("Violation LB Dual: ", i, " ", sp[num_w+num_cons+j], " ", V_L[i])
        #         push!(_E, num_w+num_cons+j)
        #         push!(r1, V_L[i])
        #     end
        # end
    end
    for (j, i) in enumerate(has_up)
        if sp[i] > X_U[i] + tol
            println("Violation UB Var: ", i, " ", sp[i], " ", X_U[i])
            push!(_E, i)
            push!(r1, X_U[i] - X[i])
        end
        # if ismin
            if sp[num_w+num_cons+num_low+j] < -tol
                println("Violation UB Dual: ", i, " ", sp[num_w+num_cons+num_low+j], " ", V_U[i])
                push!(_E, num_w+num_cons+num_low+j)
                push!(r1, V_U[i])
            end
        # else
        #     if sp[num_w+num_cons+num_low+j] > tol
        #         println("Violation UB Dual: ", i, " ", sp[num_w+num_cons+num_low+j], " ", V_U[i])
        #         push!(_E, num_w+num_cons+num_low+j)
        #         push!(r1, V_U[i])
        #     end
        # end
    end
    
    E = spzeros(num_w + num_cons + num_low + num_up, length(_E))
    for (i, j) in enumerate(_E)
        E[j, i] = 1
    end

    return E, r1
end


function compute_sensitivity(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}, 
    Δp; primal_vars=all_primal_vars(model), params=all_params(model), tol=1e-6
)
    ismin = sense_mult(primal_vars) == 1.0
    sense_multiplier = sense_mult(primal_vars)
    num_cons = length(cons)
    num_var = length(primal_vars)
    # Solution and bounds
    X, V_L, X_L, V_U, X_U, leq_locations, geq_locations, ineq_locations, has_up, has_low = compute_solution_and_bounds(primal_vars, cons)
    # Compute derivatives
    # ∂s = [∂x; ∂λ; ∂ν_L; ∂ν_U]
    ∂s, K, N = compute_derivatives_no_relax(evaluator, cons, primal_vars, params, X, V_L, X_L, V_U, X_U, leq_locations, geq_locations, ineq_locations, has_up, has_low)
    Λ = dual.(cons)
    if isnothing(Δp) || iszero(Δp)
        sp = approximate_solution(X, Λ, V_L[has_low] .* (sense_multiplier), V_U[has_up].* (-sense_multiplier), ∂s * ones(size(∂s, 2)))
        return ∂s, sp
    end
    Δs = ∂s * Δp
    sp = approximate_solution(X, Λ, V_L[has_low] .* (sense_multiplier), V_U[has_up].* (-sense_multiplier), Δs)
    # Linearly appoximated solution
    E, r1 = find_violations(X, sp, X_L, X_U, V_U, V_L, has_up, has_low, num_cons, tol)
    if !isempty(r1)
        @warn "Relaxation needed"
        Δs = fix_and_relax(E, K, N, r1, Δp, length(X), num_cons, length(has_low), sense_multiplier)
        sp = approximate_solution(X, Λ, V_L[has_low] .* (sense_multiplier), V_U[has_up].* (-sense_multiplier), Δs)
    end
    return Δs, sp
end

"""
    compute_sensitivity(model::Model; primal_vars=all_primal_vars(model), params=all_params(model))

Compute the sensitivity of the solution given sensitivity of the parameters (Δp).
"""
function compute_sensitivity(model::Model, Δp; primal_vars=all_primal_vars(model), params=all_params(model))
    evaluator, cons = create_evaluator(model; x=[primal_vars; params])
    return compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=params), evaluator, cons
end
