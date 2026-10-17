import os
import json
import base64

# Check overlay images
base_dir = "src/Tugas_UTS/6-deployment"
p_jatim = os.path.join(base_dir, "prepared_data", "jatim_lulc_overlay.png")
p_scene = os.path.join(base_dir, "prepared_data", "sentinel2_scene_overlay.png")

with open(p_jatim, "rb") as f:
    b64_jatim = base64.b64encode(f.read()).decode("utf-8")

with open(p_scene, "rb") as f:
    b64_scene = base64.b64encode(f.read()).decode("utf-8")

print(f"Jatim overlay base64 ready ({len(b64_jatim)} chars)")
print(f"Scene overlay base64 ready ({len(b64_scene)} chars)")

# Save base64 overlays as json cache so app.py doesn't re-encode every second
overlay_cache = {
    "jatim": f"data:image/png;base64,{b64_jatim}",
    "scene": f"data:image/png;base64,{b64_scene}"
}

with open(os.path.join(base_dir, "prepared_data", "overlays_cache.json"), "w", encoding="utf-8") as f:
    json.dump(overlay_cache, f)

with open(os.path.join(base_dir, "standalone_streamlit_app", "prepared_data", "overlays_cache.json"), "w", encoding="utf-8") as f:
    json.dump(overlay_cache, f)

print("Saved overlays_cache.json successfully!")
