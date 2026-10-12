import json, sys
import numpy as np
D = json.load(open(sys.argv[1]))
ms = sys.argv[2].split(",")
for e in ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]:
    c = [v for v in D.get(e, {}).values() if all(m in v for m in ms[:3])]
    print(e, len(c), {m: (round(100 * np.mean([v[m]["regret_ref"] for v in c if m in v]), 2), round(100 * np.mean([v[m]["regret"] for v in c if m in v]), 2), round(np.mean([v[m]["n_acq"] for v in c if m in v]), 2)) for m in ms if any(m in v for v in c)})
