"""
Data generator for the algebra error-type classifier (forward error injection).

Each substantive label uses 3-4 distinct injectors across varied equation templates
so the model cannot rely on a single surface pattern.
"""

import argparse
import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

LABELS = [
    "equality_balance_error",
    "negative_sign_error",
    "variable_error",
    "operation_inverse_error",
    "distribution_property_error",
    "arithmetic_slip",
    "abstain",
]

SUBSTANTIVE_LABELS = [label for label in LABELS if label != "abstain"]


# ------------------------------------------------------------------ formatting

def fmt(v):
    f = Fraction(v).limit_denominator(10000)
    if f.denominator == 1:
        return str(f.numerator)
    dec = f.numerator / f.denominator
    if abs(dec - round(dec, 2)) < 1e-9:
        return f"{dec:.2f}".rstrip("0").rstrip(".")
    return f"{f.numerator}/{f.denominator}"


def term(coef, var="x"):
    if coef == 1:
        return var
    if coef == -1:
        return f"-{var}"
    return f"{coef}{var}"


def plus_const(k):
    return f"+ {k}" if k >= 0 else f"- {abs(k)}"


def side_var_const(coef, k):
    if k == 0:
        return term(coef)
    return f"{term(coef)} {plus_const(k)}"


def slip_value(correct, rng, choices=(-2, -1, 1, 2)):
    for delta in rng.sample(list(choices), k=len(choices)):
        slipped = correct + delta
        if slipped != correct:
            return slipped
    return correct + 1


# ------------------------------------------------------------------ equation templates

def tmpl_paren(rng):
    a = rng.choice([2, 3, 4, -2, -3, 5])
    b = rng.randint(-6, 6)
    while b == 0:
        b = rng.randint(-6, 6)
    x0 = rng.randint(-6, 6)
    c = a * (x0 + b)
    problem = f"{a}(x + {b}) = {c}" if b >= 0 else f"{a}(x - {abs(b)}) = {c}"
    return problem, Fraction(x0), {"a": a, "b": b, "c": c, "kind": "paren"}


def tmpl_paren_inner_coef(rng):
    outer = rng.choice([2, 3, -2, 3, 4])
    inner = rng.choice([2, 3])
    b = rng.randint(-5, 5)
    while b == 0:
        b = rng.randint(-5, 5)
    x0 = rng.randint(-5, 6)
    c = outer * (inner * x0 + b)
    inner_term = f"{inner}x + {b}" if b >= 0 else f"{inner}x - {abs(b)}"
    problem = f"{outer}({inner_term}) = {c}"
    return problem, Fraction(x0), {
        "outer": outer,
        "inner": inner,
        "b": b,
        "c": c,
        "kind": "paren_inner",
    }


def tmpl_both_sides(rng):
    x0 = rng.randint(-6, 6)
    a = rng.choice([3, 4, 5, 6])
    c = rng.choice([1, 2, 3])
    while c == a:
        c = rng.choice([1, 2, 3])
    b = rng.randint(-9, 9)
    d = (a - c) * x0 + b
    problem = f"{side_var_const(a, b)} = {side_var_const(c, d)}"
    return problem, Fraction(x0), {"a": a, "b": b, "c": c, "d": d, "kind": "both_sides"}


def tmpl_const_first(rng):
    a = rng.choice([2, 3, 4, 5])
    b = rng.randint(1, 9)
    x0 = rng.randint(-5, 6)
    c = b + a * x0
    problem = f"{b} + {term(a)} = {c}"
    return problem, Fraction(x0), {"a": a, "b": b, "c": c, "kind": "const_first"}


def tmpl_var_const(rng):
    a = rng.choice([2, 3, 4, 5, 6])
    b = rng.randint(-9, 9)
    x0 = rng.randint(-6, 6)
    c = a * x0 + b
    problem = f"{side_var_const(a, b)} = {c}"
    return problem, Fraction(x0), {"a": a, "b": b, "c": c, "kind": "var_const"}


def tmpl_sign(rng):
    a = rng.choice([2, 3, 4, 5])
    b = rng.randint(1, 12)
    x0 = rng.randint(-5, 6)
    c = b - a * x0
    problem = f"{b} - {term(a)} = {c}"
    return problem, Fraction(x0), {"a": a, "b": b, "c": c, "kind": "sign"}


def tmpl_x_minus_const(rng):
    b = rng.randint(1, 9)
    x0 = rng.randint(-5, 8)
    c = x0 - b
    problem = f"x - {b} = {c}"
    return problem, Fraction(x0), {"b": b, "c": c, "kind": "x_minus_const"}


def tmpl_x_plus_const(rng):
    b = rng.randint(1, 9)
    x0 = rng.randint(-5, 8)
    c = x0 + b
    problem = f"x + {b} = {c}"
    return problem, Fraction(x0), {"b": b, "c": c, "kind": "x_plus_const"}


def tmpl_neg_paren(rng):
    a = rng.choice([1, 2, 3])
    b = rng.randint(1, 7)
    x0 = rng.randint(-5, 6)
    c = -a * (x0 + b)
    problem = f"-{a}(x + {b}) = {c}"
    return problem, Fraction(x0), {"a": a, "b": b, "c": c, "kind": "neg_paren"}


def tmpl_fraction(rng):
    denom = rng.choice([2, 3, 4, 5, 6, 7])
    b = rng.randint(-9, 9)
    x0 = rng.randint(-12, 12)
    while denom * x0 + b == 0:
        x0 = rng.randint(-12, 12)
    rhs = Fraction(x0, denom) + b
    if rhs.denominator == 1:
        rhs_val = rhs.numerator
        rhs_str = str(rhs_val)
    else:
        rhs_val = rhs
        rhs_str = fmt(rhs)
    problem = f"x/{denom} {plus_const(b)} = {rhs_str}"
    return problem, Fraction(x0), {
        "denom": denom,
        "b": b,
        "rhs": rhs_val,
        "kind": "fraction",
    }


def tmpl_simple_ax_eq_c(rng):
    a = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
    x0 = rng.randint(-12, 12)
    while x0 == 0:
        x0 = rng.randint(-12, 12)
    c = a * x0
    problem = f"{term(a)} = {c}"
    return problem, Fraction(x0), {"a": a, "c": c, "kind": "simple_ax"}


def tmpl_const_equals_x_plus(rng):
    a = rng.randint(4, 16)
    b = rng.randint(2, 10)
    x0 = a - b
    problem = f"{a} = x + {b}"
    return problem, Fraction(x0), {"a": a, "b": b, "kind": "const_eq_x_plus"}


# ------------------------------------------------------------------ injectors: distribution (4)

def inj_dist_partial(p, x0, rng):
    if p["kind"] != "paren":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(c - b, a)
    work = f"{term(a)} + {b} = {c}; {term(a)} = {c - b}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_dist_wrong_sign_on_constant(p, x0, rng):
    if p["kind"] != "neg_paren":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(c + b, -a)
    work = f"-{term(a)} + {b} = {c}; {term(-a)} = {c - b}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_dist_outer_only(p, x0, rng):
    if p["kind"] != "paren_inner":
        return None
    outer, inner, b, c = p["outer"], p["inner"], p["b"], p["c"]
    wrong = Fraction(c - b, outer * inner)
    work = (
        f"{outer}({inner}x + {b}) = {c}; "
        f"{outer * inner}x + {b} = {c}; "
        f"{outer * inner}x = {c - b}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_dist_skip_one_factor(p, x0, rng):
    """Distribute outer factor to the constant term only, not the variable term."""
    if p["kind"] != "paren_inner":
        return None
    outer, inner, b, c = p["outer"], p["inner"], p["b"], p["c"]
    wrong = Fraction(c - outer * b, inner)
    if wrong == x0:
        return None
    work = (
        f"{outer}({inner}x + {b}) = {c}; "
        f"{term(inner)} + {outer * b} = {c}; "
        f"{term(inner)} = {c - outer * b}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


# ------------------------------------------------------------------ injectors: balance (4)

def inj_balance_drop_const_left(p, x0, rng):
    if p["kind"] != "both_sides":
        return None
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    wrong = Fraction(d, a - c)
    work = (
        f"{side_var_const(a, b)} = {side_var_const(c, d)}; "
        f"{term(a)} = {side_var_const(c, d)}; "
        f"{term(a - c)} = {d}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_balance_add_instead_of_subtract(p, x0, rng):
    if p["kind"] != "var_const":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(c + b, a)
    work = f"{side_var_const(a, b)} = {c}; {term(a)} = {c} + {b}; {term(a)} = {c + b}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_balance_drop_variable_term(p, x0, rng):
    if p["kind"] != "both_sides":
        return None
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    wrong = Fraction(d - b, a)
    work = (
        f"{side_var_const(a, b)} = {side_var_const(c, d)}; "
        f"{side_var_const(a, b)} = {d}; "
        f"{term(a)} = {d - b}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_balance_move_const_wrong_side(p, x0, rng):
    if p["kind"] != "var_const":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(b - c, a)
    work = f"{side_var_const(a, b)} = {c}; {term(a)} = {b} - {c}; x = {fmt(wrong)}"
    return work, fmt(wrong)


# ------------------------------------------------------------------ injectors: variable (4)

def inj_var_conjoin_const(p, x0, rng):
    if p["kind"] != "const_first":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(c, a + b)
    work = f"{b} + {term(a)} = {c}; {term(a + b)} = {c}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_var_add_x_terms(p, x0, rng):
    if p["kind"] != "both_sides":
        return None
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    wrong = Fraction(d - b, a + c)
    work = (
        f"{side_var_const(a, b)} = {side_var_const(c, d)}; "
        f"{side_var_const(a + c, b)} = {d}; "
        f"{term(a + c)} = {d - b}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_var_subtract_becomes_add(p, x0, rng):
    if p["kind"] != "both_sides":
        return None
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    combined = a + c
    if combined == 0:
        return None
    wrong = Fraction(d - b, combined)
    if wrong == x0:
        return None
    work = (
        f"{side_var_const(a, b)} = {side_var_const(c, d)}; "
        f"{term(combined)} + {b} = {d}; "
        f"{term(combined)} = {d - b}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_var_merge_across_equals(p, x0, rng):
    if p["kind"] != "var_const":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    if b == 0 or a + b == 0:
        return None
    wrong = Fraction(c, a + b)
    work = f"{side_var_const(a, b)} = {c}; {term(a + b)} = {c}; x = {fmt(wrong)}"
    return work, fmt(wrong)


# ------------------------------------------------------------------ injectors: negative sign (4)

def inj_sign_drop_negative_on_coef(p, x0, rng):
    if p["kind"] != "sign":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(c - b, a)
    work = f"{b} - {term(a)} = {c}; {term(a)} = {c} - {b}; {term(a)} = {c - b}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_sign_no_flip_when_moving(p, x0, rng):
    if p["kind"] != "var_const" and p["kind"] != "x_plus_const":
        return None
    if p["kind"] == "var_const":
        a, b, c = p["a"], p["b"], p["c"]
        if b >= 0:
            return None
        wrong = Fraction(c + b, a)
        work = f"{side_var_const(a, b)} = {c}; {term(a)} = {c} + {abs(b)}; x = {fmt(wrong)}"
        return work, fmt(wrong)
    b, c = p["b"], p["c"]
    wrong = Fraction(c + b)
    work = f"x + {b} = {c}; x = {c} + {b}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_sign_reverse_subtraction_order(p, x0, rng):
    if p["kind"] != "x_minus_const":
        return None
    b, c = p["b"], p["c"]
    wrong = Fraction(c - b)
    work = f"x - {b} = {c}; x = {c} - {b}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_sign_drop_paren_negative(p, x0, rng):
    if p["kind"] != "neg_paren":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    wrong = Fraction(c, a) - b
    if wrong == x0:
        return None
    work = f"-{a}(x + {b}) = {c}; x + {b} = {Fraction(c, a)}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_sign_eq_reversal(p, x0, rng):
    """a = x + b solved as x = b - a instead of x = a - b (ex10-style sign error)."""
    if p["kind"] != "const_eq_x_plus":
        return None
    a, b = p["a"], p["b"]
    wrong = b - a
    if wrong == x0:
        return None
    work = f"x = {b} - {a}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_sign_final_drop(p, x0, rng):
    """Correct isolation through ax = k, but drop the negative on the final x (ex08-style)."""
    if p["kind"] != "var_const":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    rhs = c - b
    if x0 == 0:
        return None
    wrong = abs(x0) if x0 < 0 else -abs(x0)
    if wrong == x0:
        return None
    work = f"{side_var_const(a, b)} = {c}; {term(a)} = {rhs}; x = {fmt(wrong)}"
    return work, fmt(wrong)


# ------------------------------------------------------------------ injectors: operation inverse (4)

def inj_op_multiply_instead_of_divide(p, x0, rng):
    if p["kind"] != "const_first":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    k = c - b
    wrong = Fraction(k * a)
    work = f"{b} + {term(a)} = {c}; {term(a)} = {k}; x = {k} * {a} = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_op_fraction_wrong_inverse(p, x0, rng):
    if p["kind"] != "fraction":
        return None
    denom, b, rhs = p["denom"], p["b"], p["rhs"]
    numer = rhs - b
    wrong = Fraction(numer, denom)
    if wrong == x0:
        return None
    work = (
        f"x/{denom} {plus_const(b)} = {fmt(rhs)}; "
        f"x/{denom} = {fmt(numer)}; "
        f"x = {fmt(numer)}/{denom} = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_op_add_instead_of_divide(p, x0, rng):
    if p["kind"] != "simple_ax":
        return None
    a, c = p["a"], p["c"]
    wrong = Fraction(c + a)
    work = f"{term(a)} = {c}; x = {c} + {a} = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_op_divide_rhs_only(p, x0, rng):
    if p["kind"] != "fraction":
        return None
    denom, b, rhs = p["denom"], p["b"], p["rhs"]
    numer = rhs - b
    wrong = Fraction(numer, denom * denom)
    work = f"x/{denom} {plus_const(b)} = {fmt(rhs)}; x/{denom} = {fmt(numer)}; x = {fmt(numer)}/{denom * denom} = {fmt(wrong)}"
    return work, fmt(wrong)


# ------------------------------------------------------------------ injectors: arithmetic slip (4)

def inj_slip_subtraction_on_isolate(p, x0, rng):
    if p["kind"] != "const_first":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    k = c - b
    slip = slip_value(k, rng)
    wrong = Fraction(slip, a)
    work = f"{b} + {term(a)} = {c}; {term(a)} = {slip}; x = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_slip_balance_subtraction(p, x0, rng):
    if p["kind"] != "both_sides":
        return None
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    target = d - b
    slip = slip_value(target, rng)
    wrong = Fraction(slip, a - c)
    work = (
        f"{side_var_const(a, b)} = {side_var_const(c, d)}; "
        f"{term(a - c)} = {slip}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


def inj_slip_division_quotient(p, x0, rng):
    if p["kind"] != "simple_ax":
        return None
    a, c = p["a"], p["c"]
    slip = slip_value(c, rng, choices=(-3, -2, -1, 1, 2, 3))
    wrong = Fraction(slip, a)
    work = f"{term(a)} = {c}; x = {slip}/{a} = {fmt(wrong)}"
    return work, fmt(wrong)


def inj_slip_after_distribution(p, x0, rng):
    if p["kind"] != "paren":
        return None
    a, b, c = p["a"], p["b"], p["c"]
    expanded_rhs = c - a * b
    slip = slip_value(expanded_rhs, rng)
    wrong = Fraction(slip, a)
    work = (
        f"{a}(x + {b}) = {c}; "
        f"{term(a)} + {a * b} = {c}; "
        f"{term(a)} = {slip}; x = {fmt(wrong)}"
    )
    return work, fmt(wrong)


# Each entry: (injector_fn, [templates...], pattern_id)
INJECTORS = {
    "distribution_property_error": [
        (inj_dist_partial, [tmpl_paren], "dist_partial"),
        (inj_dist_wrong_sign_on_constant, [tmpl_neg_paren], "dist_wrong_sign_const"),
        (inj_dist_outer_only, [tmpl_paren_inner_coef], "dist_outer_only"),
        (inj_dist_skip_one_factor, [tmpl_paren_inner_coef], "dist_skip_factor"),
    ],
    "equality_balance_error": [
        (inj_balance_drop_const_left, [tmpl_both_sides], "bal_drop_const"),
        (inj_balance_add_instead_of_subtract, [tmpl_var_const], "bal_add_not_sub"),
        (inj_balance_drop_variable_term, [tmpl_both_sides], "bal_drop_x_term"),
        (inj_balance_move_const_wrong_side, [tmpl_var_const], "bal_wrong_side"),
    ],
    "variable_error": [
        (inj_var_conjoin_const, [tmpl_const_first], "var_conjoin"),
        (inj_var_add_x_terms, [tmpl_both_sides], "var_add_x"),
        (inj_var_subtract_becomes_add, [tmpl_both_sides], "var_sub_to_add"),
        (inj_var_merge_across_equals, [tmpl_var_const], "var_merge_across"),
    ],
    "negative_sign_error": [
        (inj_sign_drop_negative_on_coef, [tmpl_sign], "sign_drop_neg"),
        (inj_sign_no_flip_when_moving, [tmpl_var_const, tmpl_x_plus_const], "sign_no_flip"),
        (inj_sign_reverse_subtraction_order, [tmpl_x_minus_const], "sign_reverse_sub"),
        (inj_sign_drop_paren_negative, [tmpl_neg_paren], "sign_drop_paren_neg"),
        (inj_sign_eq_reversal, [tmpl_const_equals_x_plus], "sign_eq_reversal"),
        (inj_sign_final_drop, [tmpl_var_const], "sign_final_drop"),
    ],
    "operation_inverse_error": [
        (inj_op_multiply_instead_of_divide, [tmpl_const_first], "op_multiply"),
        (inj_op_fraction_wrong_inverse, [tmpl_fraction], "op_frac_mult"),
        (inj_op_add_instead_of_divide, [tmpl_simple_ax_eq_c], "op_add_not_div"),
        (inj_op_divide_rhs_only, [tmpl_fraction], "op_divide_rhs"),
    ],
    "arithmetic_slip": [
        (inj_slip_subtraction_on_isolate, [tmpl_const_first], "slip_sub"),
        (inj_slip_balance_subtraction, [tmpl_both_sides], "slip_balance"),
        (inj_slip_division_quotient, [tmpl_simple_ax_eq_c], "slip_div"),
        (inj_slip_after_distribution, [tmpl_paren], "slip_after_dist"),
    ],
}

ALL_PROBLEM_TMPLS = [
    tmpl_paren,
    tmpl_paren_inner_coef,
    tmpl_both_sides,
    tmpl_const_first,
    tmpl_var_const,
    tmpl_sign,
    tmpl_x_minus_const,
    tmpl_x_plus_const,
    tmpl_neg_paren,
    tmpl_fraction,
    tmpl_simple_ax_eq_c,
    tmpl_const_equals_x_plus,
]


# ------------------------------------------------------------------ generation

def generate_one(label, rng):
    specs = INJECTORS[label]
    for _ in range(50):
        inj_fn, templates, pattern_id = rng.choice(specs)
        tmpl = rng.choice(templates)
        problem, x0, params = tmpl(rng)
        out = inj_fn(params, x0, rng)
        if out is None:
            continue
        work, answer = out
        if answer == fmt(x0):
            continue
        return {
            "problem": problem,
            "correct_answer": f"x = {fmt(x0)}",
            "student_answer": f"x = {answer}",
            "student_work": work,
            "label": label,
            "pattern_id": pattern_id,
        }
    return None


def _abstain_fields(reason, label_a=None, label_b=None, grader_rationale=""):
    fields = {
        "abstain_reason": reason,
        "grader_rationale": grader_rationale,
    }
    if label_a is not None:
        fields["label_a"] = label_a
    if label_b is not None:
        fields["label_b"] = label_b
    return fields


def validate_abstain_example(ex):
    """Reject abstain rows that do not meet the human-grader bar."""
    if ex.get("label") != "abstain":
        return True

    reason = ex.get("abstain_reason")
    rationale = ex.get("grader_rationale", "").strip()
    if not reason or not rationale:
        return False

    if reason == "thin_signal_no_work":
        if ex.get("student_work") is not None:
            return False
        label_a = ex.get("label_a")
        label_b = ex.get("label_b")
        if label_a not in SUBSTANTIVE_LABELS or label_b not in SUBSTANTIVE_LABELS:
            return False
        if label_a == label_b:
            return False
        return True

    if reason == "nonsensical_work":
        work = ex.get("student_work") or ""
        steps = [part.strip() for part in work.split(";") if part.strip()]
        return len(steps) >= 3 and "label_a" not in ex and "label_b" not in ex

    if reason != "dual_label_tie":
        return False

    label_a = ex.get("label_a")
    label_b = ex.get("label_b")
    if label_a not in SUBSTANTIVE_LABELS or label_b not in SUBSTANTIVE_LABELS:
        return False
    if label_a == label_b:
        return False

    work = ex.get("student_work") or ""
    steps = [part.strip() for part in work.split(";") if part.strip()]
    if len(steps) != 3:
        return False

    pattern_id = ex.get("pattern_id")
    if pattern_id and pattern_id.startswith("abstain_dist_or_var"):
        lhs = steps[1].split("=")[0].strip()
        return "(" in steps[0] and lhs.endswith("x") and "+" not in lhs[:-1]
    if pattern_id and pattern_id.startswith("abstain_sign_or_balance"):
        # A constant crossed the equals sign without flipping its sign: the tie is
        # negative_sign_error vs equality_balance_error, both reading the same line.
        pair = {label_a, label_b}
        return (
            pair == {"negative_sign_error", "equality_balance_error"}
            and "=" in steps[0]
            and ("+" in steps[0] or "-" in steps[0])
        )
    return False


def _make_dist_or_var_abstain(problem, combined, c, x0, pattern_id, var_phrase, dist_phrase):
    """Shared builder for one-step (combined)x = c collapses from a parenthesis equation."""
    wrong = Fraction(c, combined)
    if wrong == x0 or combined <= 0:
        return None
    work = f"{problem}; {term(combined)} = {c}; x = {fmt(wrong)}"
    ex = {
        "problem": problem,
        "correct_answer": f"x = {fmt(x0)}",
        "student_answer": f"x = {fmt(wrong)}",
        "student_work": work,
        "label": "abstain",
        "pattern_id": pattern_id,
        **_abstain_fields(
            "dual_label_tie",
            "distribution_property_error",
            "variable_error",
            (
                f"One step collapses {problem} directly to {term(combined)} = {c} without "
                f"writing an expanded form. Reading A (distribution_property_error): "
                f"{dist_phrase} Reading B (variable_error): {var_phrase} "
                f"Both readings fit the same single line; neither is clearly stronger."
            ),
        ),
    }
    return ex if validate_abstain_example(ex) else None


def make_abstain_no_work(rng):
    """Final answer only: wrong value implied by add-not-subtract, two taxonomy reads tie."""
    a = rng.choice([2, 3, 4, 5, 6, 7])
    b = rng.randint(2, 20)
    x0 = rng.randint(1, 20)
    c = a * x0 + b
    wrong = Fraction(c + b, a)
    if wrong == x0:
        return None
    problem = f"{side_var_const(a, b)} = {c}"
    implied = a * wrong
    ex = {
        "problem": problem,
        "correct_answer": f"x = {fmt(x0)}",
        "student_answer": f"x = {fmt(wrong)}",
        "student_work": None,
        "label": "abstain",
        "pattern_id": "abstain_no_work",
        **_abstain_fields(
            "thin_signal_no_work",
            "equality_balance_error",
            "negative_sign_error",
            (
                f"No work shown. The answer x = {fmt(wrong)} implies {term(a)} = {fmt(implied)} "
                f"(since {fmt(wrong)} * {a} = {fmt(implied)}), which follows from adding {b} "
                f"instead of subtracting it when isolating from {problem}. Under this taxonomy, "
                f"that move is classifiable as equality_balance_error or negative_sign_error; "
                f"with no steps visible, neither label is clearly stronger."
            ),
        ),
    }
    return ex if validate_abstain_example(ex) else None


def tmpl_abstain_dist_or_var(rng):
    """One jump from a(x+b)=c to (a+b)x=c — distribute vs conjoin, single ambiguous step."""
    a = rng.choice([2, 3, 4, 5, 6])
    b = rng.randint(1, 9)
    x0 = rng.randint(1, 12)
    c = a * (x0 + b)
    combined = a + b
    problem = f"{a}(x + {b}) = {c}"
    return _make_dist_or_var_abstain(
        problem,
        combined,
        c,
        x0,
        "abstain_dist_or_var",
        f"conjoined {a} and {b} into one coefficient on x.",
        f"misapplied the distributive property on {a}(x + {b}).",
    )


def tmpl_abstain_dist_or_var_inner(rng):
    """One jump from outer(inner x + b)=c to (outer+inner)x=c."""
    outer = rng.choice([2, 3, 4])
    inner = rng.choice([2, 3, 4])
    b = rng.randint(1, 7)
    x0 = rng.randint(1, 10)
    c = outer * (inner * x0 + b)
    combined = outer + inner
    problem = f"{outer}({inner}x + {b}) = {c}"
    return _make_dist_or_var_abstain(
        problem,
        combined,
        c,
        x0,
        "abstain_dist_or_var_inner",
        f"conjoined the outer coefficient {outer} and inner x-coefficient {inner} into {combined}x.",
        f"misapplied the distributive property on {outer}({inner}x + {b}).",
    )


def tmpl_abstain_dist_or_var_sub(rng):
    """One jump from a(x-b)=c to (a-b)x=c when a>b."""
    a = rng.choice([3, 4, 5, 6, 7])
    b = rng.randint(1, min(4, a - 2))
    x0 = rng.randint(b + 1, 13)
    c = a * (x0 - b)
    combined = a - b
    if combined <= 1:
        return None
    problem = f"{a}(x - {b}) = {c}"
    return _make_dist_or_var_abstain(
        problem,
        combined,
        c,
        x0,
        "abstain_dist_or_var_sub",
        f"conjoined {a} and {b} from the x - {b} term into one coefficient on x.",
        f"misapplied the distributive property on {a}(x - {b}).",
    )


def _make_sign_or_balance_abstain(problem, a, b, c, x0, rhs_wrong, pattern_id, moved):
    """Shared builder: a constant crosses '=' without flipping sign (sign vs balance tie)."""
    wrong = Fraction(rhs_wrong, a)
    if wrong == x0:
        return None
    step1 = f"x = {rhs_wrong}" if a == 1 else f"{term(a)} = {rhs_wrong}"
    work = f"{problem}; {step1}; x = {fmt(wrong)}"
    ex = {
        "problem": problem,
        "correct_answer": f"x = {fmt(x0)}",
        "student_answer": f"x = {fmt(wrong)}",
        "student_work": work,
        "label": "abstain",
        "pattern_id": pattern_id,
        **_abstain_fields(
            "dual_label_tie",
            "negative_sign_error",
            "equality_balance_error",
            (
                f"From {problem} the student writes {step1}: {moved} "
                f"Reading A (negative_sign_error): the constant crossed the equals sign "
                f"without flipping its sign. Reading B (equality_balance_error): the constant "
                f"was changed on the right without the matching inverse operation on the left, "
                f"so the two sides were not kept balanced. Both readings explain the same "
                f"single line equally well."
            ),
        ),
    }
    return ex if validate_abstain_example(ex) else None


def tmpl_abstain_sign_or_balance_add(rng):
    """x + b = c solved as x = c + b (kept +b instead of moving it as -b)."""
    a = rng.choice([1, 2, 3, 4])
    b = rng.randint(2, 12)
    x0 = rng.randint(1, 10)
    c = a * x0 + b
    problem = f"{('x' if a == 1 else term(a))} + {b} = {c}"
    return _make_sign_or_balance_abstain(
        problem, a, b, c, x0, c + b, "abstain_sign_or_balance_add",
        f"the +{b} stayed positive on the right instead of moving across as -{b}.",
    )


def tmpl_abstain_sign_or_balance_sub(rng):
    """x - b = c solved as x = c - b (kept -b instead of moving it as +b)."""
    a = rng.choice([1, 2, 3, 4])
    b = rng.randint(2, 12)
    x0 = rng.randint(1, 10)
    c = a * x0 - b
    problem = f"{('x' if a == 1 else term(a))} - {b} = {c}"
    return _make_sign_or_balance_abstain(
        problem, a, b, c, x0, c - b, "abstain_sign_or_balance_sub",
        f"the -{b} stayed negative on the right instead of moving across as +{b}.",
    )


DUAL_LABEL_ABSTAIN_TMPLS = [
    tmpl_abstain_dist_or_var,
    tmpl_abstain_dist_or_var_inner,
    tmpl_abstain_dist_or_var_sub,
    tmpl_abstain_sign_or_balance_add,
    tmpl_abstain_sign_or_balance_sub,
]


def _rand_num(rng, lo=-18, hi=18):
    n = rng.randint(lo, hi)
    if n == 0:
        n = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    return n


def _random_work_fragment(rng):
    """Unstructured scratch step — not tied to any injector."""
    kind = rng.randint(0, 8)
    n = _rand_num(rng)
    m = _rand_num(rng)
    coef = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
    if kind == 0:
        return f"{n}x = {m}"
    if kind == 1:
        return f"x = {n} + {m}"
    if kind == 2:
        return f"{term(coef)} = {n}"
    if kind == 3:
        return f"{n} + {term(coef)} = {m}"
    if kind == 4:
        return f"x/{rng.choice([2, 3, 4, 5])} = {n}"
    if kind == 5:
        return f"{n} = x - {m}"
    if kind == 6:
        return f"{term(coef)} + {term(rng.choice([-3, -2, -1, 1, 2, 3]))} = {m}"
    if kind == 7:
        return f"x = {n} * {m}"
    return f"{n}x + {m} = {term(coef)}"


def make_abstain_nonsensical(rng):
    """Abstain: work does not fit any taxonomy bin — random unrelated steps/numbers."""
    tmpl = rng.choice(ALL_PROBLEM_TMPLS)
    problem, x0, _ = tmpl(rng)

    steps = [_random_work_fragment(rng) for _ in range(rng.randint(2, 5))]
    wrong_val = _rand_num(rng)
    while fmt(Fraction(wrong_val)) == fmt(x0):
        wrong_val = _rand_num(rng)
    steps.append(f"x = {fmt(Fraction(wrong_val))}")

    return {
        "problem": problem,
        "correct_answer": f"x = {fmt(x0)}",
        "student_answer": f"x = {fmt(Fraction(wrong_val))}",
        "student_work": "; ".join(steps),
        "label": "abstain",
        "pattern_id": "abstain_nonsensical",
        **_abstain_fields(
            "nonsensical_work",
            grader_rationale=(
                "Shown work is unstructured scratch that does not match any taxonomy "
                "bin; no single error type is supported strongly enough to label."
            ),
        ),
    }


def make_abstain_nonsensical_validated(rng):
    for _ in range(40):
        ex = make_abstain_nonsensical(rng)
        if validate_abstain_example(ex):
            return ex
    return None


def make_abstain_dual_label(rng):
    for _ in range(40):
        fn = rng.choice(DUAL_LABEL_ABSTAIN_TMPLS)
        ex = fn(rng)
        if ex is not None:
            return ex
    return None


def _try_add_example(data, seen_problems, ex, validator=None):
    """Append example only if valid and problem string is new."""
    if ex is None:
        return False
    if validator is not None and not validator(ex):
        return False
    problem = ex.get("problem")
    if not problem or problem in seen_problems:
        return False
    seen_problems.add(problem)
    data.append(ex)
    return True


def build_dataset(n, seed, abstain_frac=0.35):
    rng = random.Random(seed)
    random.seed(seed)
    data = []
    seen_problems = set()
    n_abstain = int(n * abstain_frac)
    n_abstain_no_work = n_abstain // 3
    n_abstain_nonsensical = n_abstain // 3
    n_abstain_ambiguous = n_abstain - n_abstain_no_work - n_abstain_nonsensical
    n_each = (n - n_abstain) // len(SUBSTANTIVE_LABELS)

    for label in SUBSTANTIVE_LABELS:
        made = 0
        attempts = 0
        while made < n_each and attempts < n_each * 200:
            attempts += 1
            if _try_add_example(data, seen_problems, generate_one(label, rng)):
                made += 1
        if made < n_each:
            raise RuntimeError(f"Could only generate {made}/{n_each} unique examples for {label}")

    no_work = 0
    attempts = 0
    while no_work < n_abstain_no_work and attempts < n_abstain_no_work * 200:
        attempts += 1
        if _try_add_example(data, seen_problems, make_abstain_no_work(rng), validate_abstain_example):
            no_work += 1
    if no_work < n_abstain_no_work:
        raise RuntimeError(f"Could only generate {no_work}/{n_abstain_no_work} unique no-work abstain rows")

    nonsensical = 0
    attempts = 0
    while nonsensical < n_abstain_nonsensical and attempts < n_abstain_nonsensical * 200:
        attempts += 1
        if _try_add_example(data, seen_problems, make_abstain_nonsensical_validated(rng), validate_abstain_example):
            nonsensical += 1
    if nonsensical < n_abstain_nonsensical:
        raise RuntimeError(f"Could only generate {nonsensical}/{n_abstain_nonsensical} unique nonsensical rows")

    dual_label = 0
    attempts = 0
    while dual_label < n_abstain_ambiguous and attempts < n_abstain_ambiguous * 200:
        attempts += 1
        if _try_add_example(data, seen_problems, make_abstain_dual_label(rng), validate_abstain_example):
            dual_label += 1
    if dual_label < n_abstain_ambiguous:
        raise RuntimeError(f"Could only generate {dual_label}/{n_abstain_ambiguous} unique dual-label rows")

    rng.shuffle(data)
    for i, d in enumerate(data):
        d["id"] = f"gen{i:05d}"
    return data


def print_pattern_stats(data):
    print("\nInjector pattern mix:")
    for label in SUBSTANTIVE_LABELS + ["abstain"]:
        subset = [d for d in data if d["label"] == label]
        if not subset:
            continue
        counts = Counter(d.get("pattern_id", "?") for d in subset)
        print(f"  {label}:")
        for pid, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {pid}: {n} ({n / len(subset):.0%})")


def build_abstain_review(seed=0, per_pattern=4):
    """Generate fixed samples of each abstain pattern for human review."""
    rng = random.Random(seed)
    makers = [
        ("abstain_no_work", make_abstain_no_work),
        ("abstain_nonsensical", make_abstain_nonsensical_validated),
        ("abstain_dist_or_var", tmpl_abstain_dist_or_var),
        ("abstain_dist_or_var_inner", tmpl_abstain_dist_or_var_inner),
        ("abstain_dist_or_var_sub", tmpl_abstain_dist_or_var_sub),
    ]
    rows = []
    for pattern_id, maker in makers:
        got = 0
        attempts = 0
        while got < per_pattern and attempts < 200:
            attempts += 1
            ex = maker(rng)
            if ex is None or not validate_abstain_example(ex):
                continue
            ex = dict(ex)
            ex["pattern_id"] = pattern_id
            ex["review_id"] = f"{pattern_id}_{got + 1:02d}"
            rows.append(ex)
            got += 1
        if got < per_pattern:
            raise RuntimeError(f"Could only generate {got}/{per_pattern} for {pattern_id}")
    return rows


def write_jsonl(path, rows, keep_internal=True):
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            if keep_internal:
                handle.write(json.dumps(row) + "\n")
            else:
                public = {k: v for k, v in row.items() if k != "pattern_id"}
                handle.write(json.dumps(public) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--abstain-frac", type=float, default=0.35)
    ap.add_argument(
        "--abstain-review",
        default=None,
        help="Write abstain-only review JSONL (e.g. data/abstain_review.jsonl)",
    )
    ap.add_argument(
        "--abstain-review-per-pattern",
        type=int,
        default=4,
        help="Samples per abstain pattern in review export",
    )
    args = ap.parse_args()

    if args.abstain_review:
        review = build_abstain_review(args.seed, args.abstain_review_per_pattern)
        write_jsonl(args.abstain_review, review, keep_internal=True)
        print(f"Wrote {len(review)} abstain review examples to {args.abstain_review}")
        for row in review:
            print(
                f"  {row['review_id']}: {row.get('label_a', '—')} vs {row.get('label_b', '—')} "
                f"| {row['problem']}"
            )
        return

    data = build_dataset(args.n, args.seed, args.abstain_frac)
    write_jsonl(args.out, data, keep_internal=False)

    unique_problems = len({d["problem"] for d in data})
    abstain_rows = [d for d in data if d["label"] == "abstain"]
    abstain_with_work = sum(1 for d in abstain_rows if d.get("student_work"))
    abstain_nonsensical = sum(
        1 for d in abstain_rows if d.get("pattern_id") == "abstain_nonsensical"
    )
    print(f"Wrote {len(data)} examples to {args.out}")
    print(f"Unique problems: {unique_problems}/{len(data)}")
    print("Label distribution:", dict(Counter(d["label"] for d in data)))
    print(
        f"Abstain breakdown: {len(abstain_rows)} total, "
        f"{len(abstain_rows) - abstain_with_work} no-work, "
        f"{abstain_nonsensical} nonsensical, "
        f"{abstain_with_work - abstain_nonsensical} dual-label tie"
    )
    print_pattern_stats(data)
    print("\nSample:")
    for d in data[:4]:
        print(" ", json.dumps({k: v for k, v in d.items() if k != "pattern_id"}))


if __name__ == "__main__":
    main()
