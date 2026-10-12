"""DIVA-U whose uncertainty about the elicited strengths shrinks as checks are observed, in full
acquire-then-decide episodes on the knowledge-required environments (same exact episode machinery as episodes_know.py,
test rows 0-149 as in Table tab:episodes). Schedules of the log-odds scale after n acquisitions (sigma_0 = 1, DIVA-U):
  us_sqrt: 1/sqrt(1+n),  us_inv: 1/(1+n),  us_first: 1 at the first decision, 0 afterwards (DIVA after the first step).
The schedule is selected on the development splits (rows 0-149) by the mean episode regret with the reference-posterior
final action; the test split is reported for every schedule. Writes results/f8_divaus_{dev,test}.json
({env: {tag: {policy: {regret, regret_ref, n_acq, n, ...}}}}). No LLM calls."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import episodes_know as EK
from evalx import choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u

SCHED = {"us_sqrt": lambda n: 1 / math.sqrt(1 + n), "us_inv": lambda n: 1 / (1 + n), "us_first": lambda n: 1.0 if n == 0 else 0.0}
_ps, _ev = EK.policy_step, EK.episode_value


def policy_step(kind, pol, sc, fr, kap):
    if pol not in SCHED:
        return _ps(kind, pol, sc, fr, kap)
    n = len(sc.get("observed", [])) - EK.CUR["n0"]  # checks acquired in this episode so far
    s = SCHED[pol](n)
    if kind == "tab":
        f = diva_scores(sc, fr, "cond", "greedy", "elicit", 0.75); b = f["b"]
        net = diva_u(sc, fr, s, "cond", 0.75)["net"] if s > 0 else f["net"]
    else:
        f = fio_real(sc, fr, lam=0.75); b = f["b0"]
        net = fio_real_u(sc, fr, s, 0.75)["net"] if s > 0 else f["net"]
    if not sc["checks"]:
        return "S", b
    return choose_max(net), b


def episode_value(kind, pol, sc, fr, kap):
    EK.CUR["n0"] = len(sc.get("observed", []))
    return _ev(kind, pol, sc, fr, kap)


EK.policy_step, EK.episode_value = policy_step, episode_value

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    EK.SPLIT, EK.ROW0, EK.NMAX = split, 0, 150
    EK.POLS = (["never", "diva", "diva_u"] if split == "test" else ["diva_u"]) + list(SCHED)
    EK.OUTNAME = f"f8_divaus_{split}"; EK.SKIP_TAGS = set()
    os.environ.setdefault("EPK_PROC", "64")
    sys.argv = sys.argv[:1]
    EK.main()
