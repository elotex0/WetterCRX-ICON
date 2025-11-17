import os
import sys
import json
from datetime import datetime

png_root = sys.argv[1]  # z.B. icond2/<RUN>
run = sys.argv[2]
date = sys.argv[3] if len(sys.argv) > 3 else datetime.utcnow().strftime("%Y%m%d")

metadata = {
    "run": run,
    "date": date,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "var_types": [],
    "timesteps": {}
}

for var_type in os.listdir(png_root):
    var_path = os.path.join(png_root, var_type)
    if not os.path.isdir(var_path):
        continue

    # var_type merken
    metadata["var_types"].append(var_type)

    # PNG-Dateien finden
    files = sorted(f for f in os.listdir(var_path) if f.endswith(".png"))
    timesteps = []

    for f in files:
        # Beispiel: t2m_20251008_0700.png → 20251008_0700
        parts = f.replace(".png", "").split("_")
        if len(parts) >= 2:
            timestep = "_".join(parts[-2:]) if len(parts[-1]) == 4 else parts[-1]
            timesteps.append(timestep)

    metadata["timesteps"][var_type] = timesteps

# metadata.json liegt **außerhalb** des run-Ordners
meta_path = os.path.join(os.path.dirname(png_root), "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Metadata written to {meta_path}")
