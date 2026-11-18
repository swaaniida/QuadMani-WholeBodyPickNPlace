# OBB branch replay analysis

Source recording: `20261113T124258_247402Z_2de9b2`

Analyzed captures: 5 / 5

Each frame uses the same filtered camera-frame object point cloud for all branches. All candidates retained.

Uniform comparison confidence is the official discriminator re-evaluation; original diffusion confidence is separately saved.

Safe/contact-like/penetrating describe final OPEN CAD versus estimated plane; no approach-path, reachability or grasp-success guarantee.

Camera OBB approaches +camera Z. World OBB approaches -support-plane normal; yaw samples full 360 degrees and offsets preserve the official five values.

Minimum-region labels are local-Z heuristics; the merged visual OBJ has no reliable anatomical part labels.

Shallow ranking = penetration ascending, confidence descending. Least-penetration highlight breaks ties with maximum clearance; representative breaks ties with confidence. Highlights may coincide.


## Aggregate observations (this recording only)

- diffusion: safe 1/1000; mean penetration 93.74 mm. Highest-confidence grasp penetrates in 5/5 captures.

- obb_camera: safe 104/250; mean penetration 12.40 mm. Highest-confidence grasp penetrates in 5/5 captures.

- obb_world: safe 50/250; mean penetration 25.17 mm. Highest-confidence grasp penetrates in 5/5 captures.

Highest measured safe fraction: obb_camera. This does not establish grasp success; OBB positions and incomplete visible geometry must also be inspected.


## Selection implications

Use safe fraction, confidence among safe candidates, clearance distribution and opening-volume occupancy/contact evidence together. Zero penetration alone can favor a grasp floating above the object. Temporal and trajectory safety are not evaluated here.


## Errors
