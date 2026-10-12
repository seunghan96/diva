"""Unit tests with a mock transport (no network, no tokenizer). Run from naacl_dcfr/infoseek:
    python -m pytest -q baselines/tests
"""
import asyncio, gzip, json, os, re, subprocess, sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines import common as C, policies as P, episode as E, llm as L  # noqa: E402
from baselines.toy_env import make_scenario  # noqa: E402


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def letters_in(prompt):
    return re.findall(r"^\(([A-Z])\) ", prompt, flags=re.M)


def bin_letter(prompt, value):
    """Letter of the 11-bin option equal to value (handles ascending/descending order)."""
    for L_, v in re.findall(r"^\(([A-K])\) (\d+)%$", prompt, flags=re.M):
        if int(v) == round(value * 100):
            return L_
    raise AssertionError("no bin")


def make_llm(fn, keep="all"):
    return L.LLM("mock", L.MockTransport(fn), render=L.plain_render, log=L.CallLog(keep_prompts=keep))


SCN = make_scenario(3)


def user_part(prompt):
    return prompt.split("[USER]", 1)[1].split("[ASSISTANT]")[0]


def assistant_part(prompt):
    return prompt.split("[ASSISTANT]", 1)[1]


# ------------------------------------------------------------------ parsers
def test_parse_json_variants():
    assert C.parse_json('blah {"A": 0.1, "B": -0.2}') == {"A": 0.1, "B": -0.2}
    assert C.parse_json("```json\n{'A': 0.1, 'B': 2,}\n```") == {"A": 0.1, "B": 2}
    assert C.parse_json('<think>{"x":1}</think> final {"A": 30%}')["A"] == pytest.approx(0.3)
    assert C.parse_json('first {"a":1} then {"b":2}') == {"b": 2}
    assert C.parse_json("no json here") is None
    assert C.parse_json('nested {"a": {"b": 1}}') == {"a": {"b": 1}}


def test_strip_reasoning():
    assert C.strip_reasoning("<|channel|>analysis<|message|>xx<|end|><|channel|>final<|message|>ANSWER: B") == "ANSWER: B"
    assert C.strip_reasoning("<think>ANSWER: A</think>ANSWER: C") == "ANSWER: C"


def test_parse_choice():
    d2o = {"A": "C", "B": "A", "S": "S"}  # display -> original (permuted)
    assert C.parse_choice("... so ANSWER: B", d2o) == ("A", "explicit")
    assert C.parse_choice("Final answer: (A)", d2o) == ("C", "explicit")
    assert C.parse_choice("I would run (B).", d2o) == ("A", "paren")
    assert C.parse_choice("We have enough; stop and decide now.", d2o)[0] == "S"
    assert C.parse_choice("ANSWER: Z", d2o)[0] is None
    assert C.parse_choice("I want the Forward guidance check", d2o, {"C": "Forward guidance", "A": "x"})[0] == "C"
    assert C.parse_choice("**ANSWER:** S", d2o)[0] == "S"


def test_to_float():
    assert C.to_float("0.25 points") == 0.25
    assert C.to_float("30%") == pytest.approx(0.3)
    assert C.to_float(True) is None


# ------------------------------------------------------------------ rendering
def test_options_permutation_and_bodies():
    st = C.State(scn=dict(SCN))
    rem = st.remaining
    txt, d2o = C.options_text(st, perm=rem[::-1])
    assert d2o["A"] == rem[-1] and d2o["S"] == "S"
    assert txt.count("\n(") == len(rem) + 1  # rem options + S
    tp = {k: v for k, v in SCN["text_parts"].items() if k != "option_lines"}
    scn = dict(SCN, text_parts=dict(tp, options="(A) Custom one\n  continued\n(B) Custom two"))
    assert C.option_body(scn, "A") == "Custom one continued"
    assert "shows whether" in C.option_body(scn, "C")  # fallback rendering


def test_evidence_includes_results_and_hypothetical():
    st = C.State(scn=SCN, acquired=[{"label": "A", "value": 1}])
    t = C.evidence_text(st, hypothetical=("B", 0))
    assert "Result of (A)" in t and "(Hypothetical)" in t and "did NOT find" in t and "Already known" in t
    assert "A" not in st.remaining


# ------------------------------------------------------------------ belief / decision
def test_belief_two_orders_average():
    def fn(prompt, kind, **kw):
        return {bin_letter(prompt, 0.3): 1.0}
    llm = make_llm(fn)
    b = run(C.belief(llm, C.State(scn=SCN)))
    assert b == pytest.approx(0.3)
    assert len(llm.log.rows) == 2 and all(r["ok"] for r in llm.log.rows)


def test_belief_none_when_no_logprobs():
    llm = make_llm(lambda p, k, **kw: None)
    assert run(C.belief(llm, C.State(scn=SCN))) is None
    assert llm.log.rows[0]["err"]


def test_final_decision_debiased():
    d = C.domain_info(SCN)

    def fn(prompt, kind, **kw):
        u = user_part(prompt)
        if "probability that" in u:
            return {bin_letter(u, 0.2): 1.0}
        # model prefers BUY regardless of position, with letter-A bias
        m = dict(re.findall(r"^\(([AB])\) (\w+)$", u, flags=re.M))
        buy = [k for k, v in m.items() if v == "BUY"][0]
        return {buy: 0.7, ("B" if buy == "A" else "A"): 0.3}
    out = run(C.final_decision(make_llm(fn), C.State(scn=SCN)))
    assert out["action"] == "BUY" and out["probs"]["BUY"] == pytest.approx(0.7) and out["belief"] == pytest.approx(0.2)
    assert out["prob"] == pytest.approx(0.7)


# ------------------------------------------------------------------ policies
def prefer_label(target):
    """Mock: letter calls put mass on whatever display letter maps to the original `target` check name."""
    name = {c["label"]: c["name"] for c in SCN["checks"]}.get(target)

    def fn(prompt, kind, **kw):
        u = user_part(prompt)
        if "probability that" in u:
            return {bin_letter(u, 0.5): 1.0}
        opts = dict(re.findall(r"^\(([A-Z])\) (.*)$", u, flags=re.M))
        if target == "S" and "S" in opts:
            return {k: (0.9 if k == "S" else 0.1 / len(opts)) for k in opts}
        for k, v in opts.items():
            if name and v.startswith(name):
                return {kk: (0.8 if kk == k else 0.2 / (len(opts) - 1)) for kk in opts}
        return {k: 1 / len(opts) for k in opts}
    return fn


@pytest.mark.parametrize("cls", [P.DirectChoice, P.Salience])
def test_letter_policies_follow_preference_under_permutation(cls):
    llm = make_llm(prefer_label("C"))
    out = run(cls().step(llm, C.State(scn=SCN)))
    assert out["choice"] == "C"
    assert out["scores"]["C"] == pytest.approx(0.8, abs=0.05)
    assert len([r for r in llm.log.rows if r["kind"] == "letters"]) == 2  # two permutations


def test_direct_stop():
    out = run(P.DirectChoice().step(make_llm(prefer_label("S")), C.State(scn=SCN)))
    assert out["choice"] == "S"


def test_salience_budget():
    st = C.State(scn=SCN, acquired=[{"label": "A", "value": 0}])
    out = run(P.Salience(budget=1).step(make_llm(prefer_label("C")), st))
    assert out["choice"] == "S"


def test_uncertainty_only():
    # belief 0.5 -> entropy 1 bit -> acquire; belief 0.0 -> stop
    out = run(P.UncertaintyOnly(tau=0.5).step(make_llm(prefer_label("B")), C.State(scn=SCN)))
    assert out["choice"] == "B" and out["info"]["entropy"] == pytest.approx(1.0)

    def sure(prompt, kind, **kw):
        return {bin_letter(user_part(prompt), 0.0): 1.0}
    out = run(P.UncertaintyOnly(tau=0.5).step(make_llm(sure), C.State(scn=SCN)))
    assert out["choice"] == "S"


def test_finrisk_ask():
    def fn(prompt, kind, **kw):
        u = user_part(prompt)
        if "PROCEED" in u:
            m = dict(re.findall(r"^\(([AB])\) (\w+):", u, flags=re.M))
            ask = [k for k, v in m.items() if v == "ASK"][0]
            return {ask: 0.8, ("A" if ask == "B" else "B"): 0.2}
        return prefer_label("D")(prompt, kind)
    out = run(P.FinRiskAsk().step(make_llm(fn), C.State(scn=SCN)))
    assert out["choice"] == "D" and out["info"]["p_ask"] == pytest.approx(0.8)
    assert out["scores"]["S"] == pytest.approx(0.2)


def imagined_mock(strong):
    """b0=0.5; a hypothetical result of check `strong` moves belief to 0.9/0.1; others stay at 0.5."""
    nm = {c["label"]: c["name"] for c in SCN["checks"]}[strong]

    def fn(prompt, kind, **kw):
        u = user_part(prompt)
        b = 0.5
        if "(Hypothetical) Suppose " + nm in u:
            b = 0.1 if "did NOT" in u else 0.9
        if "Assume it is known" in u:  # BED-LLM likelihoods
            if "If " + nm + " were performed now" in u:
                b = 0.1 if "NOT the case" in u else 0.9
            else:
                b = 0.5
        return {bin_letter(u, b): 1.0}
    return fn


def test_eig_imagined_picks_informative_and_stops():
    out = run(P.EIGImagined(tau=0.05).step(make_llm(imagined_mock("E")), C.State(scn=SCN)))
    assert out["choice"] == "E" and out["scores"]["E"] > 0.5
    assert all(abs(out["scores"][k]) < 1e-6 for k in "ABCDF")

    def flat(prompt, kind, **kw):
        return {bin_letter(user_part(prompt), 0.5): 1.0}
    assert run(P.EIGImagined(tau=0.05).step(make_llm(flat), C.State(scn=SCN)))["choice"] == "S"


def test_bed_llm():
    out = run(P.BEDLLM().step(make_llm(imagined_mock("B")), C.State(scn=SCN)))
    assert out["choice"] == "B"
    # H(0.5) - H(0.9) with b0 = 0.5
    assert out["scores"]["B"] == pytest.approx(1 - C.H([0.9, 0.1]), abs=1e-6)


def test_voi_prompt_parse_retry_and_stop():
    calls = {"n": 0}

    def fn(prompt, kind, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "I think check B is best."  # unparseable -> retry
        return 'Reasoning... {"A": -0.1, "B": 0.4, "C": "0.1 points", "D": 0, "E": 0, "F": 0}'
    llm = make_llm(fn)
    out = run(P.VoIPrompt().step(llm, C.State(scn=SCN)))
    assert out["choice"] == "B" and out["scores"]["C"] == pytest.approx(0.1)
    assert len(llm.log.rows) == 2 and llm.log.rows[0]["raw"][0].startswith("I think")

    out = run(P.VoIPrompt().step(make_llm(lambda p, k, **kw: '{"A": -1, "B": -0.2}'), C.State(scn=SCN)))
    assert out["choice"] == "S"
    out = run(P.VoIPrompt().step(make_llm(lambda p, k, **kw: "garbage"), C.State(scn=SCN)))
    assert out["choice"] == "S" and out["info"]["fail"] == "parse"


def test_voi_paper_pipeline():
    d = C.domain_info(SCN)
    pos, neg = d.get("adverse_short", "adverse"), d.get("benign_short", "not adverse")
    nmC = {c["label"]: c["name"] for c in SCN["checks"]}["C"]
    g, l = SCN["g"], SCN["l"]

    def fn(prompt, kind, **kw):
        u = user_part(prompt)
        if '"finding"' in u:
            return '{"finding": 0.5, "no finding": 0.5}'
        b = 0.5
        if "(Hypothetical) Suppose " + nmC in u:
            b = 0.05 if "did NOT" in u else 0.95
        return json.dumps({pos: b, neg: 1 - b})
    out = run(P.VoIPaper().step(make_llm(fn), C.State(scn=SCN)))
    def vstar(b):
        return max((1 - b) * g - b * l, 0.0)
    expect = 0.5 * vstar(0.95) + 0.5 * vstar(0.05) - vstar(0.5)
    assert out["info"]["sim"]["C"]["voi"] == pytest.approx(expect)
    assert out["scores"]["C"] == pytest.approx(expect - SCN["checks"][2]["cost"])
    assert out["choice"] == ("C" if expect - SCN["checks"][2]["cost"] > 0 else "S")


def test_cot_need_parse_and_fallback():
    out = run(P.CoTNeed().step(make_llm(lambda p, k, **kw: "Step 1 ... Step 2 ...\nANSWER: D"), C.State(scn=SCN)))
    assert out["choice"] == "D" and out["info"]["parse"] == "explicit"
    llm = make_llm(lambda p, k, **kw: "hmm")
    out = run(P.CoTNeed().step(llm, C.State(scn=SCN)))
    assert out["choice"] == "S" and out["info"]["parse"] == "fail_default_stop"
    assert len(llm.log.rows) == 3 and all(r["raw"] == ["hmm"] for r in llm.log.rows)


def test_sc_cot_votes():
    ans = ["ANSWER: B", "ANSWER: B", "ANSWER: S", "nonsense", "ANSWER: C"]
    out = run(P.SelfConsistencyCoT(k=5).step(make_llm(lambda p, k, i=0, **kw: ans[i]), C.State(scn=SCN)))
    assert out["choice"] == "B" and out["scores"]["B"] == pytest.approx(0.5) and out["info"]["parse_fail"] == 1


def test_react_episode():
    d = C.domain_info(SCN)

    def fn(prompt, kind, **kw):
        u = user_part(prompt)
        if kind == "letters":
            if "probability that" in u:
                return {bin_letter(u, 0.3): 1.0}
            return {"A": 0.6, "B": 0.4}
        a = assistant_part(prompt)
        if "Observation:" not in a:
            return " I should look at margins.\nAction: check[B]\nObservation: (hallucinated)"
        return " Good enough.\nAction: decide[BUY]"
    pol = P.ReAct()
    llm = make_llm(fn)
    row = run(E.episode(pol, llm, SCN))
    assert [x["label"] for x in row["acquired"]] == ["B"]
    assert row["decision"] == "BUY" and row["decision_info"]["react_action"] == "BUY"
    assert row["traj"][-1]["choice"] == "S"
    assert "(hallucinated)" not in json.dumps(row, default=str)
    assert "eval" in row and "policy_regret" in row["eval"]


def test_react_invalid_then_fallback():
    llm = make_llm(lambda p, k, **kw: {"A": .5, "B": .5} if k == "letters" else " hmm\nAction: check[Z]")
    out = run(P.ReAct().step(llm, C.State(scn=SCN)))
    assert out["choice"] == "S" and out["info"]["fail"] == "parse_default_stop" and len(out["info"]["bad"]) == 3


def test_random_deterministic():
    a = run(P.RandomPolicy().step(None, C.State(scn=SCN)))  # noqa
    b = run(P.RandomPolicy().step(None, C.State(scn=SCN)))
    assert a == b and a["choice"] in C.State(scn=SCN).remaining + ["S"]


def test_oracle_matches_exact_gold():
    for i in range(30):
        scn = make_scenario(i)
        out = run(P.make_policy("oracle_voi").step(None, C.State(scn=scn)))
        assert out["choice"] == scn["exact"]["gold_choice"]
        dp = run(P.make_policy("oracle_dp").step(None, C.State(scn=scn)))
        assert dp["choice"] in C.State(scn=scn).remaining + ["S"]


def test_oracle_dp_matches_exact():
    for i in range(30):
        scn = make_scenario(i)
        out = run(P.make_policy("oracle_dp").step(None, C.State(scn=scn)))
        assert out["choice"] == scn["exact"]["dp_first_action"] or out["choice"] in scn["exact"]["dp_first_ties"]
        r = run(E.episode(P.make_policy("oracle_dp"), None, scn))
        assert r["eval"]["policy_regret"] == pytest.approx(0.0, abs=1e-6)  # DP policy has zero regret


def test_always_all_and_never_episodes():
    llm = make_llm(lambda p, k, **kw: {"A": 0.5, "B": 0.5} if k == "letters" else "")
    r = run(E.episode(P.AlwaysAll(), llm, SCN))
    assert r["n_acq"] == len(SCN["checks"]) and r["cost"] == pytest.approx(sum(c["cost"] for c in SCN["checks"]))
    r = run(E.episode(P.Never(), llm, SCN))
    assert r["n_acq"] == 0 and r["cost"] == 0


def test_world_uses_stored_episodes():
    y, vals = E.world(SCN, 1)
    assert y == SCN["episodes"][1]["state"] and vals == SCN["episodes"][1]["outcomes"]


def test_registry_complete():
    need = {"random", "salience", "uncertainty", "eig", "bed_llm", "voi_prompt", "voi_paper", "finrisk_ask", "react",
            "cot_need", "sc_cot", "direct", "always_all", "never", "oracle_voi", "oracle_dp"}
    assert need <= set(P.REGISTRY)


def test_cli_llm_free(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmd = [sys.executable, "-m", "baselines.run", "--model", "none", "--methods", "all", "--toy", "5",
           "--mode", "both", "--out", str(tmp_path)]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True)
    rows = [json.loads(x) for x in gzip.open(tmp_path / "none" / "oracle_voi" / "toy.jsonl.gz", "rt")]
    assert len(rows) == 5 and all("error" not in r for r in rows)
    ep = [json.loads(x) for x in gzip.open(tmp_path / "none" / "oracle_dp" / "toy.episode.jsonl.gz", "rt")]
    assert len(ep) == 5 and all(abs(r["eval"]["policy_regret"]) < 1e-6 for r in ep)
    subprocess.run(cmd, cwd=root, check=True, capture_output=True)  # resume: no duplicates
    rows = [json.loads(x) for x in gzip.open(tmp_path / "none" / "oracle_voi" / "toy.jsonl.gz", "rt")]
    assert len(rows) == 5


# ------------------------------------------------------------------ real env rows (all three tracks), generic mock
import glob  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rows(pattern, n=3):
    fs = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not fs:
        return []
    out = []
    with gzip.open(fs[0], "rt") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


ENV_ROWS = (_rows("envs_synth/data/binary/finance_pilot*.jsonl.gz") + _rows("envs_synth/data/ms/finance_pilot*.jsonl.gz")
            + _rows("envs_real/data/*_dev.jsonl.gz"))


def generic_mock(prompt, kind, i=0, **kw):
    """Plausible outputs for any prompt: letter calls -> mildly non-uniform over the listed letters;
    JSON prompts -> JSON over the requested keys; CoT -> ANSWER line; ReAct -> check then decide."""
    u = user_part(prompt)
    if kind == "letters":
        ls = re.findall(r"^\(([A-Z])\) ", u, flags=re.M)
        return {L_: (j + 1.0) for j, L_ in enumerate(ls)} if ls else None
    m = re.search(r"keys exactly \[(.*?)\]", u)
    if m:
        keys = json.loads("[" + m.group(1) + "]")
        return json.dumps({k: 1.0 / len(keys) for k in keys})
    if "STRICT JSON mapping every check letter" in u:
        ls = [x for x in re.findall(r"^\(([A-Z])\) ", u, flags=re.M)]
        return json.dumps({k: (0.1 if j == 0 else -0.1) for j, k in enumerate(ls)})
    if "Thought" in u:
        a = assistant_part(prompt)
        acts = re.findall(r"decide\[([A-Z]+)\]", u)
        return " ok\nAction: check[A]" if "Observation:" not in a else f" done\nAction: decide[{acts[0]}]"
    return "reasoning...\nANSWER: B" if i % 2 == 0 else "ANSWER: S"


@pytest.mark.skipif(not ENV_ROWS, reason="env data not built yet")
@pytest.mark.parametrize("name", sorted(P.REGISTRY))
def test_all_policies_on_env_rows(name):
    for scn in ENV_ROWS:
        scn = dict(scn)
        pol = P.make_policy(name)
        llm = make_llm(generic_mock, keep="free")
        st = C.State(scn=scn)
        out = run(pol.step(llm, st))
        assert out["choice"] in st.remaining + ["S"], (scn["id"], out)
        assert "fail" not in out.get("info", {}), (scn["id"], out["info"])
        fs = run(E.first_step(P.make_policy(name), make_llm(generic_mock), scn))
        assert "rank_err" not in fs, fs
        ep = run(E.episode(P.make_policy(name), make_llm(generic_mock), scn))
        assert ep["decision"] in scn["actions"], (scn["id"], ep["decision_info"])
        assert "eval" in ep, ep.get("eval_err")
        assert ep["n_acq"] <= scn["budget"]


@pytest.mark.skipif(not ENV_ROWS, reason="env data not built yet")
def test_oracle_voi_matches_gold_on_env_rows():
    for scn in ENV_ROWS:
        out = run(P.make_policy("oracle_voi").step(None, C.State(scn=dict(scn))))
        assert out["choice"] == scn["exact"]["gold_choice"], scn["id"]


@pytest.mark.skipif(not ENV_ROWS, reason="env data not built yet")
def test_rendering_env_rows():
    for scn in ENV_ROWS:
        st = C.State(scn=dict(scn))
        txt, d2o = C.options_text(st, perm=st.remaining[::-1])
        assert len(d2o) == len(st.remaining) + 1
        lab = st.remaining[0]
        st.acquired.append({"label": lab, "value": int(scn["episodes"][0]["outcomes"][lab])})
        ev = C.evidence_text(st, hypothetical=(st.remaining[0], 0))
        assert "Result of (" + lab + ")" in ev and "(Hypothetical)" in ev
        assert lab not in C.options_text(st)[1].values()
        assert "_meta_do_not_show" not in ev and "ticker" not in ev.lower()
