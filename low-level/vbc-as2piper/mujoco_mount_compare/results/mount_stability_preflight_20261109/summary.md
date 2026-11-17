# Standing/impulse preflight

This is not a walking stability test and not the official AS2 controller. It checks model integrity and relative standing response only. Do not use it to certify walking.

30 N × 50 ms = 1.5 N·s horizontal impulse; identical initial joint pose and PD hold on all three models. These are preliminary static support tests, not walking.

| Variant | Condition | Max roll (deg) | Max pitch (deg) | Min feet contact | Fall screen |
|---|---|---:|---:|---:|---:|
| as2_only | no_impulse | 0.02 | 0.70 | 4 | no |
| as2_only | forward | 0.02 | 0.70 | 4 | no |
| as2_only | backward | 0.02 | 0.70 | 4 | no |
| as2_only | left | 0.50 | 0.70 | 4 | no |
| as2_only | right | 0.55 | 0.70 | 4 | no |
| as2_piper_center | no_impulse | 0.03 | 0.88 | 4 | no |
| as2_piper_center | forward | 0.03 | 0.88 | 4 | no |
| as2_piper_center | backward | 0.03 | 0.99 | 4 | no |
| as2_piper_center | left | 0.50 | 0.88 | 4 | no |
| as2_piper_center | right | 0.55 | 0.88 | 4 | no |
| as2_piper_head_selected | no_impulse | 0.03 | 2.12 | 4 | no |
| as2_piper_head_selected | forward | 0.03 | 2.38 | 4 | no |
| as2_piper_head_selected | backward | 0.03 | 2.13 | 4 | no |
| as2_piper_head_selected | left | 0.43 | 2.12 | 4 | no |
| as2_piper_head_selected | right | 0.48 | 2.12 | 4 | no |
