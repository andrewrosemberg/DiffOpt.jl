using MathOptInterface
import MathOptInterface: Evaluator, _forward_eval_ϵ
const MOI = MathOptInterface
using MathOptInterface.Nonlinear

mutable struct MOI.Evaluator{B} <: MOI.AbstractMathOptInterface.NLPEvaluator
    # The internal datastructure.
    model::Model
    # The abstract-differentiation backend
    backend::B
    # ordered_constraints is needed because `OrderedDict` doesn't support
    # looking up a key by the linear index.
    ordered_constraints::Vector{ConstraintIndex}
    # Storage for the NLPBlockDual, so that we can query the dual of individual
    # constraints without needing to query the full vector each time.
    constraint_dual::Vector{Float64}
    # Timers
    initialize_timer::Float64
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_objective_timer::Float64
    eval_hessian_constraint_timer::Float64
    eval_hessian_lagrangian_timer::Float64
    parameters_as_variables::Bool

    function MOI.Evaluator(
        model::Model,
        backend::B = nothing,
    ) where {B<:Union{Nothing,MOI.AbstractMathOptInterface.NLPEvaluator}}
        return new{B}(
            model,
            backend,
            ConstraintIndex[],
            Float64[],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            false
        )
    end
end

function Evaluator{B}(
    # The internal datastructure.
    model::Model,
    # The abstract-differentiation backend
    backend::B,
    # ordered_constraints is needed because `OrderedDict` doesn't support
    # looking up a key by the linear index.
    ordered_constraints::Vector{ConstraintIndex},
    # Storage for the NLPBlockDual, so that we can query the dual of individual
    # constraints without needing to query the full vector each time.
    constraint_dual::Vector{Float64},
    # Timers
    initialize_timer::Float64,
    eval_objective_timer::Float64,
    eval_constraint_timer::Float64,
    eval_objective_gradient_timer::Float64,
    eval_constraint_gradient_timer::Float64,
    eval_constraint_jacobian_timer::Float64,
    eval_hessian_objective_timer::Float64,
    eval_hessian_constraint_timer::Float64,
    eval_hessian_lagrangian_timer::Float64,
) where {B<:Union{Nothing,MOI.AbstractMathOptInterface.NLPEvaluator}}
    return Evaluator{B}(
        # The internal datastructure.
        model::Model,
        # The abstract-differentiation backend
        backend::B,
        # ordered_constraints is needed because `OrderedDict` doesn't support
        # looking up a key by the linear index.
        ordered_constraints::Vector{ConstraintIndex},
        # Storage for the NLPBlockDual, so that we can query the dual of individual
        # constraints without needing to query the full vector each time.
        constraint_dual::Vector{Float64},
        # Timers
        initialize_timer::Float64,
        eval_objective_timer::Float64,
        eval_constraint_timer::Float64,
        eval_objective_gradient_timer::Float64,
        eval_constraint_gradient_timer::Float64,
        eval_constraint_jacobian_timer::Float64,
        eval_hessian_objective_timer::Float64,
        eval_hessian_constraint_timer::Float64,
        eval_hessian_lagrangian_timer::Float64,
        false
    )
end


function MOI.Nonlinear.ReverseAD._forward_eval_ϵ(
    d::MathOptInterface.Nonlinear.NLPEvaluator,
    ex::Union{_FunctionStorage,_SubexpressionStorage},
    storage_ϵ::AbstractVector{ForwardDiff.Partials{N,T}},
    partials_storage_ϵ::AbstractVector{ForwardDiff.Partials{N,T}},
    x_values_ϵ,
    subexpression_values_ϵ,
    user_operators::Nonlinear.OperatorRegistry,
) where {N,T}
    @assert length(storage_ϵ) >= length(ex.nodes)
    @assert length(partials_storage_ϵ) >= length(ex.nodes)
    zero_ϵ = zero(ForwardDiff.Partials{N,T})
    # ex.nodes is already in order such that parents always appear before children
    # so a backwards pass through ex.nodes is a forward pass through the tree
    children_arr = SparseArrays.rowvals(ex.adj)
    for k in length(ex.nodes):-1:1
        node = ex.nodes[k]
        partials_storage_ϵ[k] = zero_ϵ
        if node.type == Nonlinear.NODE_VARIABLE
            storage_ϵ[k] = x_values_ϵ[node.index]
        elseif node.type == Nonlinear.NODE_VALUE
            storage_ϵ[k] = zero_ϵ
        elseif node.type == Nonlinear.NODE_SUBEXPRESSION
            storage_ϵ[k] = subexpression_values_ϵ[node.index]
        elseif node.type == Nonlinear.NODE_PARAMETER
            storage_ϵ[k] = x_values_ϵ[node.index]
        # elseif !(d.parameters_as_variables) && node.type == Nonlinear.NODE_PARAMETER
        #     storage_ϵ[k] = zero_ϵ
        else
            @assert node.type != Nonlinear.NODE_MOI_VARIABLE
            ϵtmp = zero_ϵ
            @inbounds children_idx = SparseArrays.nzrange(ex.adj, k)
            for c_idx in children_idx
                @inbounds ix = children_arr[c_idx]
                @inbounds partial = ex.partials_storage[ix]
                @inbounds storage_val = storage_ϵ[ix]
                # TODO: This "if" statement can take 8% of the hessian
                # evaluation time. Find a more efficient way.
                if !isfinite(partial) && storage_val == zero_ϵ
                    continue
                end
                ϵtmp += storage_val * ex.partials_storage[ix]
            end
            storage_ϵ[k] = ϵtmp
            if node.type == Nonlinear.NODE_CALL_MULTIVARIATE
                # TODO(odow): consider how to refactor this into Nonlinear.
                op = node.index
                n_children = length(children_idx)
                if op == 3 # :*
                    # Lazy approach for now.
                    anyzero = false
                    tmp_prod = one(ForwardDiff.Dual{TAG,T,N})
                    for c_idx in children_idx
                        ix = children_arr[c_idx]
                        sval = ex.forward_storage[ix]
                        gnum = ForwardDiff.Dual{TAG}(sval, storage_ϵ[ix])
                        tmp_prod *= gnum
                        anyzero = ifelse(sval * sval == zero(T), true, anyzero)
                    end
                    # By a quirk of floating-point numbers, we can have
                    # anyzero == true && ForwardDiff.value(tmp_prod) != zero(T)
                    if anyzero || n_children <= 2
                        for c_idx in children_idx
                            prod_others = one(ForwardDiff.Dual{TAG,T,N})
                            for c_idx2 in children_idx
                                (c_idx == c_idx2) && continue
                                ix = children_arr[c_idx2]
                                gnum = ForwardDiff.Dual{TAG}(
                                    ex.forward_storage[ix],
                                    storage_ϵ[ix],
                                )
                                prod_others *= gnum
                            end
                            partials_storage_ϵ[children_arr[c_idx]] =
                                ForwardDiff.partials(prod_others)
                        end
                    else
                        for c_idx in children_idx
                            ix = children_arr[c_idx]
                            prod_others =
                                tmp_prod / ForwardDiff.Dual{TAG}(
                                    ex.forward_storage[ix],
                                    storage_ϵ[ix],
                                )
                            partials_storage_ϵ[ix] =
                                ForwardDiff.partials(prod_others)
                        end
                    end
                elseif op == 4 # :^
                    @assert n_children == 2
                    idx1 = first(children_idx)
                    idx2 = last(children_idx)
                    @inbounds ix1 = children_arr[idx1]
                    @inbounds ix2 = children_arr[idx2]
                    @inbounds base = ex.forward_storage[ix1]
                    @inbounds base_ϵ = storage_ϵ[ix1]
                    @inbounds exponent = ex.forward_storage[ix2]
                    @inbounds exponent_ϵ = storage_ϵ[ix2]
                    base_gnum = ForwardDiff.Dual{TAG}(base, base_ϵ)
                    exponent_gnum = ForwardDiff.Dual{TAG}(exponent, exponent_ϵ)
                    if exponent == 2
                        partials_storage_ϵ[ix1] = 2 * base_ϵ
                    elseif exponent == 1
                        partials_storage_ϵ[ix1] = zero_ϵ
                    else
                        partials_storage_ϵ[ix1] = ForwardDiff.partials(
                            exponent_gnum * pow(base_gnum, exponent_gnum - 1),
                        )
                    end
                    result_gnum = ForwardDiff.Dual{TAG}(
                        ex.forward_storage[k],
                        storage_ϵ[k],
                    )
                    # TODO(odow): fix me to use NaNMath.jl instead
                    log_base_gnum = base_gnum < 0 ? NaN : log(base_gnum)
                    partials_storage_ϵ[ix2] =
                        ForwardDiff.partials(result_gnum * log_base_gnum)
                elseif op == 5 # :/
                    @assert n_children == 2
                    idx1 = first(children_idx)
                    idx2 = last(children_idx)
                    @inbounds ix1 = children_arr[idx1]
                    @inbounds ix2 = children_arr[idx2]
                    @inbounds numerator = ex.forward_storage[ix1]
                    @inbounds numerator_ϵ = storage_ϵ[ix1]
                    @inbounds denominator = ex.forward_storage[ix2]
                    @inbounds denominator_ϵ = storage_ϵ[ix2]
                    recip_denominator =
                        1 / ForwardDiff.Dual{TAG}(denominator, denominator_ϵ)
                    partials_storage_ϵ[ix1] =
                        ForwardDiff.partials(recip_denominator)
                    partials_storage_ϵ[ix2] = ForwardDiff.partials(
                        -ForwardDiff.Dual{TAG}(numerator, numerator_ϵ) *
                        recip_denominator *
                        recip_denominator,
                    )
                elseif op > 6
                    f_input = _UnsafeVectorView(d.jac_storage, n_children)
                    for (i, c) in enumerate(children_idx)
                        f_input[i] = ex.forward_storage[children_arr[c]]
                    end
                    H = _UnsafeLowerTriangularMatrixView(
                        d.user_output_buffer,
                        n_children,
                    )
                    has_hessian = Nonlinear.eval_multivariate_hessian(
                        user_operators,
                        user_operators.multivariate_operators[node.index],
                        H,
                        f_input,
                    )
                    # This might be `false` if we extend this code to all
                    # multivariate functions.
                    @assert has_hessian
                    for col in 1:n_children
                        dual = zero(ForwardDiff.Partials{N,T})
                        for row in 1:n_children
                            # Make sure we get the lower-triangular component.
                            h = row >= col ? H[row, col] : H[col, row]
                            # Performance optimization: hessians can be quite
                            # sparse
                            if !iszero(h)
                                i = children_arr[children_idx[row]]
                                dual += h * storage_ϵ[i]
                            end
                        end
                        i = children_arr[children_idx[col]]
                        partials_storage_ϵ[i] = dual
                    end
                end
            elseif node.type == Nonlinear.NODE_CALL_UNIVARIATE
                @inbounds child_idx = children_arr[ex.adj.colptr[k]]
                f′′ = Nonlinear.eval_univariate_hessian(
                    user_operators,
                    user_operators.univariate_operators[node.index],
                    ex.forward_storage[child_idx],
                )
                partials_storage_ϵ[child_idx] = f′′ * storage_ϵ[child_idx]
            end
        end
    end
    return storage_ϵ[1]
end