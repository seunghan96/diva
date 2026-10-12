"""baselines.run with the prior-work policies of prior_policies.py registered (dellma, uot); same CLI and outputs."""
import asyncio
from baselines import policies as P, run as R
from baselines.prior_policies import PRIOR
from baselines.voi_tool import VoITool

P.REGISTRY.update(PRIOR)
P.REGISTRY["voi_tool"] = VoITool

if __name__ == "__main__":
    asyncio.run(R.main())
