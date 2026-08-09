# Tomato hand 5D viewer

Local, hand-only URDF viewer for the current tomato-picking five-dimensional
G20 manifold. The browser never connects to SDK, CAN, or hardware. Every 5D
slider request is mapped by the production Python function
`g20_synergy_to_joint_targets_v2`; the front end does not maintain a duplicate
kinematic formula.

Run from the repository root:

```bash
cd tools/tomato_hand_5d_viewer && npm install && cd ../..
python script/serve_tomato_hand_5d_viewer.py --open
```

The launcher opens
`http://127.0.0.1:8765/?mapping=v2&load=latest`. The pose selector provides the production
pregrasp center plus the negative and positive endpoint of each of the five
continuous manifold axes (11 inspection poses total). Each pose keeps its own
temporary 5D values and per-joint overrides while the page remains open.

The legacy Clean-v5 mapping remains available at `http://127.0.0.1:8765/`
for comparison only. Physics-v1 imports the calibrated v2 mapper directly.

The frozen source draft is `logs/hand_5d_viewer/latest_draft.json`; its SHA-256
is recorded by `simulation/config/tomato_hand_5d_user_calibrated_v2.json` and
verified by the focused tests. Saving a new draft never changes the production
manifold, simulation config, or training config automatically.
