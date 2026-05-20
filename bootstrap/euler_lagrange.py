"""
Euler-Lagrange derivative and integration by parts.

The EL derivative of a Lagrangian density L by a field φ is:
    δL/δφ = ∂L/∂φ - ∂_μ(∂L/∂φ_{,μ}) + ∂_μ∂_ν(∂L/∂φ_{,μν})

Since we limit ourselves to at most 2 derivatives, the series terminates.

Also provides:
- remove_second_derivatives: integration by parts to put Lagrangians
  in first-order form (no second derivatives of fields)
"""

from sympy import S, Rational
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr, Tensor
from bootstrap.tensor_algebra import (
    Lorentz, metric, h, dh, ddh, fresh_indices, canon, _JET_HIERARCHY
)
from bootstrap.jet import jet_derivative, total_derivative, _decompose_tensmul, _sum_terms


def euler_lagrange(lagrangian, wrt_head):
    """Compute the Euler-Lagrange derivative δL/δ(field).
    
    For a field with indices (like h_{μν}), the result has the same
    free indices. These are generated as fresh contravariant indices.
    
    δL/δh_{αβ} = ∂L/∂h_{αβ} - ∂_γ(∂L/∂h_{αβ,γ}) + ∂_γ∂_δ(∂L/∂h_{αβ,γδ})
    
    Args:
        lagrangian: scalar tensor expression (the Lagrangian density)
        wrt_head: TensorHead to take the EL derivative by (e.g., h, phi)
        
    Returns:
        (result, el_indices): 
            result is the tensor expression for δL/δ(field)
            el_indices is the tuple of free indices on the result
    """
    info = _JET_HIERARCHY.get(wrt_head)
    if info is None:
        raise ValueError(f"Unknown field {wrt_head} — not in jet hierarchy")
    
    n_field = info['n_field_indices']
    child = info.get('child')  # first derivative head
    grandchild_info = _JET_HIERARCHY.get(child, {}) if child else {}
    grandchild = grandchild_info.get('child')  # second derivative head
    
    # Generate fresh indices for the EL derivative result
    # These are the "free" indices of δL/δφ_{αβ...}
    el_indices = fresh_indices(n_field) if n_field > 0 else ()
    
    # Term 1: ∂L/∂φ_{α...}
    term1 = jet_derivative(lagrangian, wrt_head, list(el_indices))
    
    # Term 2: -∂_γ(∂L/∂φ_{α...,γ})
    term2 = S.Zero
    if child is not None:
        gamma, = fresh_indices(1)
        child_indices = list(el_indices) + [gamma]
        dL_dchild = jet_derivative(lagrangian, child, child_indices)
        if dL_dchild != S.Zero:
            term2 = -total_derivative(dL_dchild, -gamma)
    
    # Term 3: +∂_γ∂_δ(∂L/∂φ_{α...,γδ})
    term3 = S.Zero
    if grandchild is not None:
        gamma2, delta = fresh_indices(2)
        gc_indices = list(el_indices) + [gamma2, delta]
        dL_dgc = jet_derivative(lagrangian, grandchild, gc_indices)
        if dL_dgc != S.Zero:
            # Apply two total derivatives
            first_deriv = total_derivative(dL_dgc, -delta)
            if first_deriv != S.Zero:
                term3 = total_derivative(first_deriv, -gamma2)
    
    result = term1 + term2 + term3
    
    # Canonicalize the result
    if isinstance(result, TensExpr):
        result = canon(result)
    
    return result, el_indices


def euler_lagrange_scalar(lagrangian, wrt_head):
    """Euler-Lagrange derivative for a scalar field (no free indices).
    
    Convenience wrapper around euler_lagrange for scalar fields.
    Returns just the expression (no indices to track).
    """
    result, _ = euler_lagrange(lagrangian, wrt_head)
    return result


def remove_second_derivatives(lagrangian, heads_to_process=None):
    """Integration by parts: remove second derivatives from a Lagrangian.
    
    Uses the jet derivative identity:
        L = (dL/ddh_{mnrs}) * ddh_{mnrs} + terms without ddh
    to isolate the ddh part, then replaces it with:
        -(d/dx^s)(dL/ddh_{mnrs}) * dh_{mnr} + boundary
    
    This avoids the fragile index-manipulation approach.
    
    Args:
        lagrangian: tensor expression for the Lagrangian
        heads_to_process: optional list of second-derivative TensorHeads.
            
    Returns:
        Modified Lagrangian with no second derivatives.
    """
    if heads_to_process is None:
        heads_to_process = [ddh]
        for head, info in _JET_HIERARCHY.items():
            if hasattr(head, 'name') and str(head.name).startswith('dd'):
                if head != ddh and head not in heads_to_process:
                    heads_to_process.append(head)
    
    result = lagrangian
    for dd_head in heads_to_process:
        result = _ibp_via_jet_deriv(result, dd_head)
    
    return canon(result) if isinstance(result, TensExpr) else result


def _ibp_via_jet_deriv(expr, dd_head):
    """Remove dd_head from expr by integration by parts.
    
    For each term containing dd_head, we use the identity:
        f(...) * ddh_{μνρσ} → -∂_σ(f(...)) * dh_{μνρ}
    
    We process each term of expr individually to avoid index clashes.
    """
    from bootstrap.jet import _get_component, _get_indices
    
    info = _JET_HIERARCHY.get(dd_head)
    if info is None or 'parent' not in info:
        return expr
    parent_head = info['parent']
    
    # Check if dd_head appears at all
    test_indices = fresh_indices(len(dd_head.index_types))
    dL_test = jet_derivative(expr, dd_head, list(test_indices))
    if dL_test == S.Zero:
        return expr  # no dd_head present
    
    # Process each term of expr individually
    if isinstance(expr, TensAdd):
        terms = list(expr.args)
    else:
        terms = [expr]
    
    out_terms = []
    for term in terms:
        coeff, factors = _decompose_tensmul(term) if isinstance(term, TensMul) else (
            S.One, [term] if isinstance(term, Tensor) else []
        )

        if not factors:
            out_terms.append(term)  # scalar, no ddh
            continue

        # Find ddh factor in this term
        dd_idx = None
        for i, f in enumerate(factors):
            if _get_component(f) == dd_head:
                dd_idx = i
                break

        if dd_idx is None:
            out_terms.append(term)  # no ddh in this term
            continue
        
        # Extract the ddh factor and its indices
        dd_factor = factors[dd_idx]
        dd_indices = _get_indices(dd_factor)
        other_factors = factors[:dd_idx] + factors[dd_idx + 1:]

        # The last index of ddh is the one we integrate by parts on.
        # In a scalar Lagrangian term this index is a dummy paired with an
        # opposite-sign occurrence elsewhere in the term. When we hand the
        # IBP index to total_derivative, it gets appended to each other
        # factor (Leibniz); if that factor already has the same index with
        # the same sign, sympy rejects the build with "two equal covariant
        # indices". The fix: rename this dummy pair to a fresh name in
        # dd_factor and in the unique other-factor that contains the
        # partner. (substitute_indices is no help here — it operates on
        # free indices only and is a silent no-op on dummies.)
        last = dd_indices[-1]
        partner = -last
        fresh_base, = fresh_indices(1)
        new_last = fresh_base if last.is_up else -fresh_base
        new_partner = -new_last
        new_other_factors = []
        for f in other_factors:
            f_indices = _get_indices(f)
            if partner in f_indices:
                comp = _get_component(f)
                new_indices = [new_partner if i == partner else i
                               for i in f_indices]
                new_other_factors.append(comp(*new_indices))
            else:
                new_other_factors.append(f)
        other_factors = new_other_factors
        # Rebuild dd_factor with the renamed last index. If the partner
        # was inside ddh itself (rare, but it's allowed — e.g. a ddh trace),
        # we rename there too.
        dd_indices_new = []
        for j, idx in enumerate(dd_indices):
            if j == len(dd_indices) - 1:
                dd_indices_new.append(new_last)
            elif idx == partner:
                dd_indices_new.append(new_partner)
            else:
                dd_indices_new.append(idx)
        dd_factor = dd_head(*dd_indices_new)
        dd_indices = dd_indices_new
        parent_indices = dd_indices[:-1]
        ibp_index = dd_indices[-1]

        # Build: -coeff * ∂_{ibp_index}(other_factors) * dh(parent_indices)
        # The derivative acts on the "other" factors via total_derivative
        other_expr = coeff
        for f in other_factors:
            other_expr = other_expr * f
        
        if other_expr == S.Zero:
            continue
        
        # Take total derivative of the other factors
        if isinstance(other_expr, TensExpr):
            d_other = total_derivative(other_expr, ibp_index)
        elif other_expr == coeff:  # pure number, derivative is zero
            d_other = S.Zero
        else:
            d_other = S.Zero
        
        # Build parent: dh with the parent indices
        parent = parent_head(*parent_indices)
        
        if d_other == S.Zero:
            # total_derivative of constant * other is zero only if
            # other has no field factors. But there should be field 
            # factors (h, dh) for this to contribute. Add the constant
            # times parent as the ibp result (which comes from ∂(constant)=0):
            # Actually if d_other = 0, the IBP'd term vanishes.
            # But we still removed the ddh term! Need to add back only 
            # the non-ddh part. Wait - let me reconsider.
            # If d_other = 0, then ∂_σ(f) = 0, so -∂_σ(f) * dh = 0.
            # The ddh term was: coeff * f * ddh. After IBP: 0.
            # So we just drop this term entirely.
            continue
        
        # Build the IBP'd term: -d_other * parent
        # Distribute over d_other's terms to avoid index clashes
        if isinstance(d_other, TensAdd):
            for dt in d_other.args:
                out_terms.append(-dt * parent)
        else:
            out_terms.append(-d_other * parent)

    result = _sum_terms(out_terms)
    return canon(result) if isinstance(result, TensExpr) else result
