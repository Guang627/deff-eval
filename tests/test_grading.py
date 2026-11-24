import pytest

from passk_smooth.data.grading import extract_answer, grade, is_equiv, normalize


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so the answer is \\boxed{5}.", "5"),
        ("first \\boxed{1} then \\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
        ("nested \\boxed{\\sqrt{\\frac{1}{2}}}", "\\sqrt{\\frac{1}{2}}"),
        ("\\fbox{7}", "7"),
        ("The answer is 42.", "42"),
        ("no answer here", None),
    ],
)
def test_extract(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "pred,gold",
    [
        ("\\dfrac{3}{4}", "\\frac{3}{4}"),
        ("3/4", "\\frac34"),
        ("0.75", "\\frac{3}{4}"),
        ("x = 3", "3"),
        ("1,000", "1000"),
        ("90^\\circ", "90"),
        ("\\text{(B)}", "(B)"),
        ("5.0", "5"),
        ("\\left(1,2\\right)", "(1,2)"),
    ],
)
def test_equivalent(pred, gold):
    assert is_equiv(pred, gold, use_math_verify=False)


@pytest.mark.parametrize("pred,gold", [("4", "5"), ("\\frac{1}{2}", "\\frac{1}{3}"), (None, "1")])
def test_not_equivalent(pred, gold):
    assert not is_equiv(pred, gold, use_math_verify=False)


def test_grade_end_to_end():
    assert grade("Reasoning... \\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}", use_math_verify=False)
    assert not grade("Reasoning... \\boxed{2}", "3", use_math_verify=False)


def test_normalize_idempotent():
    for s in ["\\frac{1}{2}", "x=5", "1,000", "\\sqrt2"]:
        assert normalize(normalize(s)) == normalize(s)


def test_gsm_symbolic_records(tmp_path):
    import json

    from passk_smooth.data.problems import gsm_final_answer, load_problems

    assert gsm_final_answer("so 3 + 4 = 7\n#### 1,250") == "1250"
    rows = [
        {"id": t, "instance": i, "question": f"Q{t}-{i}", "answer": f"... #### {t + i}", "original_id": 100 + t, "config": cfg}
        for cfg in ("main", "p1")
        for t in range(3)
        for i in range(5)
    ]
    path = tmp_path / "gsm.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    ps = load_problems(str(path), variants_per_template=2)
    assert len(ps) == 2 * 3 * 2
    p = next(q for q in ps if q["problem_id"] == "p1/t002/i01")
    assert p["subject"] == "t002" and p["level"] == "p1" and p["answer"] == "3" and p["source"] == "102"
