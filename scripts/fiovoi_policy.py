"""FIO-VOI as a baselines-compatible Policy (step/decide), driven by a cached factored elicitation (run_factored.py).
Because the LLM-specified model is scenario-level, the whole sequential policy needs no further LLM calls.
Variants: mode='myopic' (acquire argmax VOI_hat - cost while > tau), 'dp' (exact DP on the elicited model),
'eig' (acquire argmax EIG on the elicited model while > eig_tau), 'budget' (forced k acquisitions by VOI order)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dcore import posterior, voi_exact, eig_exact, dp_value, best_action, dp_factory

STOP = "S"


def clip(x):
    return min(max(float(x), 0.01), 0.99)


class FIOVOI:
    def __init__(self, fac_by_id, mode="myopic", tau=0.0, eig_tau=0.05, name=None):
        self.fac = fac_by_id
        self.mode = mode
        self.tau = tau
        self.eig_tau = eig_tau
        self.name = name or f"fiovoi_{mode}"

    def _model(self, st):
        fr = self.fac[st.scn["id"]]
        prior = clip(fr["prior"])
        obs = [(clip(o[0]), clip(o[1]), ob["value"]) for o, ob in zip(fr["obs"], st.scn.get("observed", []))]
        acq = [(clip(fr["chk"][a["label"]][0]), clip(fr["chk"][a["label"]][1]), a["value"]) for a in st.acquired]
        b = posterior(prior, obs + acq)
        rem = {k: (clip(fr["chk"][k][0]), clip(fr["chk"][k][1]), st.cost(k)) for k in st.remaining}
        return b, rem

    async def step(self, llm, st):
        b, rem = self._model(st)
        g, l = float(st.scn["g"]), float(st.scn["l"])
        if not rem:
            return {"choice": STOP, "scores": {}}
        voi = {k: voi_exact(b, s1, s0, g, l) - c for k, (s1, s0, c) in rem.items()}
        eig = {k: eig_exact(b, s1, s0) for k, (s1, s0, c) in rem.items()}
        if self.mode == "myopic":
            k = max(voi, key=voi.get)
            return {"choice": k if voi[k] > self.tau else STOP, "scores": voi, "info": {"b": b}}
        if self.mode == "dp":
            if len(st.scn["checks"]) > 8:  # DP intractable -> myopic (flagged)
                k = max(voi, key=voi.get)
                return {"choice": k if voi[k] > self.tau else STOP, "scores": voi, "info": {"b": b, "dp_fallback": 1}}
            sid = st.scn["id"]
            if not hasattr(self, "_V"):
                self._V = {}
            if sid not in self._V:
                fr = self.fac[sid]
                par = {c["label"]: (clip(fr["chk"][c["label"]][0]), clip(fr["chk"][c["label"]][1]), float(c.get("cost", 0))) for c in st.scn["checks"]}
                self._V = {sid: dp_factory(par, g, l)}  # keep only the current scenario's cache
            v, a = self._V[sid](b, tuple(rem))
            return {"choice": a, "scores": voi, "info": {"b": b, "dp": v}}
        if self.mode == "eig":
            k = max(eig, key=eig.get)
            return {"choice": k if eig[k] > self.eig_tau else STOP, "scores": eig, "info": {"b": b}}
        if self.mode == "budget":  # never stops voluntarily; ranking by raw VOI then EIG as tie-break
            k = max(rem, key=lambda x: (round(voi[x] + rem[x][2], 9), eig[x]))
            return {"choice": k, "scores": voi, "info": {"b": b}}
        raise ValueError(self.mode)

    async def decide(self, llm, st):
        b, _ = self._model(st)
        g, l = float(st.scn["g"]), float(st.scn["l"])
        a = best_action(b, g, l)
        acts = st.scn.get("actions") or ["act", "pass"]
        return {"action": acts[0] if a == "act" else acts[1], "belief": b}
