<img src="./fig2.png" width="400px"></img>

## Rigidformer

Implementation of [RigidFormer](https://arxiv.org/abs/2605.09196), Learning Rigid Dynamics using Transformers, out of MIT and Meta

## Install

```bash
$ pip install rigidformer
```

## Usage

```python
import torch
from rigidformer import Rigidformer

# instantiate model

model = Rigidformer(
    dim = 768,
    dim_head = 128,
    arope_dim = 96,
    arope_base = 10_000,
    heads = 6,
    dropout = 0.1,
    object_self_attn_depth = 4,
    anchor_cross_attn_depth = 4,
    num_register_tokens = 16,
    num_anchors = 4,
    object_hidden_layers = (0, 1, 2, 4),
    vertex_properties_dim = 3,
    pointnet_vertex_dim = 1024,
    pointnet_ratios = (1., .5, .25, .125),
    pointnet_local_mlp_depth = 4,
    anchor_query_hidden_dim = 2048,
    anchor_query_hidden_depth = 2,
    anchor_predictor_ff_depth = 6
)

# mock inputs

physical_dt = torch.full((2,), 1. / 60.)
step_code = torch.ones(2)
mass = torch.rand(2, 4) + 0.1
friction = torch.rand(2, 4)
restitution = torch.rand(2, 4)
vertex_properties = torch.stack((mass, friction, restitution), dim = -1)
# exact paper order: [m, mu, epsilon] = [mass, friction, restitution]
object_first_frame_pos = torch.randn(2, 4, 64, 3)
object_pos = torch.randn(2, 4, 64, 3)       # (batch, num_objects, num_points, 3)
object_pos_prev = torch.randn(2, 4, 64, 3)
object_pos_next = torch.randn(2, 4, 64, 3)

# training

loss, loss_breakdown = model(
    physical_dt = physical_dt,
    step_code = step_code,
    vertex_properties = vertex_properties,
    object_pos = object_pos,
    object_pos_prev = object_pos_prev,
    object_first_frame_pos = object_first_frame_pos,
    object_pos_next = object_pos_next  # target
)

loss.backward()

# if `object_pos_next` not passed in, will return predictions

pred = model(
    physical_dt = physical_dt,
    step_code = step_code,
    vertex_properties = vertex_properties,
    object_pos = object_pos,
    object_pos_prev = object_pos_prev,
    object_first_frame_pos = object_first_frame_pos,
)

assert pred.object_pos_next.shape == object_pos.shape

# rollout multiple steps with a wrapper

from rigidformer import RigidformerRolloutWrapper

wrapper = RigidformerRolloutWrapper(model)

rollout_positions = wrapper(
    num_steps = 10,
    physical_dt = physical_dt,
    step_code = step_code,
    vertex_properties = vertex_properties,
    object_positions = [object_pos_prev, object_pos]
)

# rollout_positions is a list of length 12 tensors of shape (batch, num_objects, num_points, 3)
# includes the 2 initial positions
```

### Parameter-matched inferred profile

The paper reports 174.8M trainable parameters but does not publish enough
internal PointNet and anchor-predictor widths or depths to derive that count.
The default configuration therefore uses an explicit, academically conventional
capacity inference while preserving every disclosed outer dimension: `D=768`,
four object-decoder layers, four decoder feature scales, six 128D heads, and
2.5x SwiGLU expansion.

- The shared PointNet backbone follows progressive widths
  `256 -> 512 -> 1024 -> 1024 -> 1024`, and each of the three local hierarchy
  MLPs has four pointwise convolution layers.
- The 271D anchor input is projected by `271 -> 2048 -> 2048 -> 768`, where
  2048 is the conventional `8/3 * D` hidden width.
- Each of the four paper feature scales uses exactly one independent pre-norm
  gated cross-attention block. Each block then uses six residual
  FiLM-plus-2.5x-SwiGLU refinement layers to match the reported capacity. That
  MLP depth and predictor FiLM are explicit reproduction assumptions because
  the paper does not disclose the block internals.
- Object self-attention is pre-normalized with RMSNorm in addition to the
  paper's per-head QK normalization.

This profile has exactly **175,259,905 trainable parameters**, 459,905 (0.26%)
above the paper's 174.8M report. It is intentionally labeled inferred rather
than official until the authors publish their model code or checkpoint.

`physical_dt` and `step_code` intentionally have different meanings and must
not be substituted for one another. `physical_dt` is the elapsed simulator
time in seconds and its square is used for Verlet integration and loss
normalization. `step_code` is the dimensionless paper code `s` (normally one
of `{1, 5, 10}`); FiLM receives exactly `(s, s^2)`. The per-vertex motion
feature remains the paper's discrete displacement `x_t - x_{t-1}`, not that
displacement divided by `physical_dt`.

### Paper T=8 training protocol

The sequence trainer uses eight sampled states per item: two observed warmup
frames followed by six supervised autoregressive predictions. A single step
code `s` is sampled near-uniformly from `{1, 5, 10}` and held fixed across the
whole window; `physical_dt = base_physical_dt * s` is computed separately.
One shuffled training-scene visit produces one fresh random window, so every
trajectory contributes exactly one item per epoch. The random start is uniform
over all valid starts for the sampled `s`.
Predictions are fed back from the first predicted frame onward,
the first frame remains the rigid reference, FPS anchors are reused across all
six steps, and the four-term anchor loss is averaged over time. Rollout inputs
remain predicted, while Eq. 11 acceleration targets always use three
ground-truth states from the sampled sequence.

Each object's three physical attributes use the fixed paper order
`[m, mu, epsilon]`: mass, coefficient of friction, and coefficient of
restitution. `vertex_properties` therefore has shape `(batch, objects, 3)`
(or `(batch, objects, points, 3)` when already expanded per vertex). The
Box2D generator reads these values from the created body and fixture; an Isaac
Sim adapter must preserve the same order.

The two Appendix E augmentations are enabled in their disclosed locations.
Before collation, each training sample permutes all objects with probability
`0.5`, using the same permutation for trajectory positions, velocities,
physical properties, masks, and cached FPS/anchor data. After a batch reaches
the model device, the sequence wrapper rotates the entire batch about the
Z-axis with probability `0.5`. One angle is sampled from
`{5 degrees, 10 degrees, ..., 355 degrees}` and shared by every object and all
eight frames, so the finite-difference acceleration targets are computed from
the consistently augmented trajectory. Calling `eval()` disables rotation.
`Box2DSequenceDataset` performs the pre-collation permutation automatically.
An Isaac Sim or other custom `Dataset.__getitem__` should pass its sample
dictionary through `apply_rigidformer_object_permutation_augmentation` before
returning it; dataset-specific first-axis object tensors can be listed with
`additional_object_tensor_keys`.

```python
from rigidformer import (
    RigidformerSequenceTrainingWrapper,
    RigidformerTrainingConfig,
    build_rigidformer_optimizer_and_scheduler,
    rigidformer_training_step,
    sample_rigidformer_training_windows
)

# native-rate simulation data: (batch, frames, objects, points, xyz)
trajectories = torch.randn(18, 80, 4, 64, 3)
window = sample_rigidformer_training_windows(
    trajectories,
    base_physical_dt = 1. / 60.
)

trainer = RigidformerSequenceTrainingWrapper(model)
config = RigidformerTrainingConfig()
optimizer, scheduler = build_rigidformer_optimizer_and_scheduler(
    trainer,
    steps_per_epoch = 100,
    config = config
)

output, gradient_norm = rigidformer_training_step(
    trainer,
    dict(
        object_positions = window.object_positions,
        physical_dt = window.physical_dt,
        step_code = window.step_code,
        # fixed last-axis order: [mass, friction, restitution]
        vertex_properties = torch.rand(18, 4, 3)
    ),
    optimizer,
    gradient_clip_norm = config.gradient_clip_norm,
    scheduler = scheduler
)
```

The paper states that training uses eight frames and no scheduled sampling,
but the authors have not released the training loop. Treating those eight
frames as two warmup states plus a six-step closed-loop objective is therefore
an explicit reproduction choice, not a claim about unavailable author code.

The default hierarchical PointNet follows the dimensions disclosed in the paper: a 1024-channel per-vertex Conv1d backbone, four geometry scales (100%, 50%, 25%, and 12.5%), and fusion to the object-token width. The paper does not disclose the intermediate Conv1d widths or KNN neighborhood sizes; those are explicit configurable reproduction assumptions in this implementation. Hierarchical neighborhood lookup uses exact device-native KNN: CUDA inputs run `bmm + topk` entirely on the GPU while streaming supports in bounded chunks, and CPU execution is retained only as a development/test fallback. The reference-frame point cloud is required because the final rigid projection aligns reference anchors and scatters the resulting transform to reference vertices.

The main configuration uses the paper's 96D ARoPE inside each 128D attention head: 32 rotary channels per spatial axis and 32 pass-through channels. The 16 register tokens receive zero rotary phase and are therefore unpositioned. The paper specifies log-spaced frequencies but does not disclose their base; `arope_base = 10_000` is the conventional RoPE reproduction assumption. Reduced toy models must set `arope_dim` explicitly to a positive multiple of six that does not exceed `dim_head`.

The paper's `dropout = 0.1` is applied to attention probabilities in every
object and anchor attention module and after each SwiGLU output projection.

### 300-epoch distributed training

The production entry point uses the paper's 300 epochs, main-model dimensions,
batch size 18 per process, AdamW schedule, gradient clipping, T=8 closed-loop
objective, and both augmentations by default. Launch one process per GPU with
`torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 -m rigidformer.train \
    --train-data /data2/jinruixing/mfs/rigidformer/datasets/RigidFormer-MOVi-A/isaac_movi.h5 \
    --output-dir /data2/jinruixing/mfs/rigidformer/runs/main_300e \
    --micro-batch-size-per-process 2 \
    --amp bf16 \
    --resume auto
```

`--amp bf16` is a practical A100/L40 training choice, not a precision mode
specified by the paper. The rigid Kabsch/SVD projection is always evaluated in
FP32 and its rollout result is cast back to the model dtype. Use `--amp none`
when a strict full-FP32 numerical baseline is preferred.

`--batch-size-per-process` remains the paper's effective value 18. On a 46 GiB
L40, `--micro-batch-size-per-process 2` accumulates nine micro-batches before
each optimizer step. DDP synchronization is deferred to the final micro-batch,
and accumulated numerator gradients are normalized by the global count of
valid objects. This is mathematically the same global masked mean as a direct
batch of 18 per process; it is not a smaller-batch approximation.

On the current eight-L40 server, NCCL peer-to-peer/IB discovery hangs during
process-group initialization, so launch with the validated transport fallback:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --standalone \
    --nproc-per-node=8 -m rigidformer.train [arguments above]
```

The trainer binds every NCCL process group to `cuda:LOCAL_RANK` and applies a
10-minute initialization/collective timeout, so transport failures terminate
with an error instead of waiting indefinitely.

The MOVi epoch follows the disclosed 960-scene training split and the
scene-as-item convention of the referenced HopNet data pipeline: the scene list
is shuffled, every scene is visited once, and each visit samples one random
T=8 window. The previous arbitrary 16-fold trajectory repetition has been
removed. The final incomplete batch is retained; discarding it would silently
change both scene coverage and the learning-rate schedule. With 960 scenes,
8 processes, and effective batch size 18 per process, each rank receives 120
scenes and runs 7 optimizer steps per epoch (six groups of 18 plus one group
of 12), for 2,100 steps over 300 epochs and 70 warmup steps. At L40
micro-batch 2 this is 60 forward/backward micro-batches per rank and epoch,
grouped in nines. The run header records both counts, and checkpoints refuse
resume when the sampling, micro-batch, accumulation, or `steps_per_epoch`
protocol differs.

The paper authors' repository currently lists MOVi training code as forthcoming,
and the paper does not disclose the process count for its main MOVi run.
Consequently, the scene-epoch definition above removes the known 16x mismatch,
but an exact author optimizer-step comparison still requires their released
training code and original world size.

The Isaac-MOVi-A HDF5 reader is lazy and opens one read-only handle per data
loader worker. It validates `rigidformer-isaac-hdf5-v1`, reconstructs world
points as `R(quaternion_wxyz) @ points_local + translation_world`, preserves
the cached four FPS anchors, and passes dynamic object/point lengths through
collation. Physical attributes are read in the schema-mandated
`[mass_kg, dynamic_friction, restitution]` order. Crucially,
`physical_dt = record_dt_s * s`; `base_physics_dt_s` is only the PhysX internal
substep and is never passed to the model or FiLM. For this supplied archive the
stored 960 training IDs are unchanged, the first 120 IDs in the frozen stored
validation order remain validation, and the final 120 become test. This yields
the required 960/120/120 counts without data-dependent or random re-splitting.
Because these are newly generated Isaac scenes, this matches the experimental
split protocol and counts, not the unavailable identities of the paper's
original MOVi-A scenes.

The `.npz` archive contains padded `positions` with shape
`(trajectories, frames, max_objects, max_points, 3)`, `props` with shape
`(trajectories, max_objects, 3)` in `[mass, friction, restitution]` order,
integer `object_lens` with shape `(trajectories,)`, and integer `point_lens`
with shape `(trajectories, max_objects)`. Valid objects and points must occupy
prefixes. Every valid object has a point length in `[4, max_points]`; padded
object entries in `point_lens` must be exactly zero. Object and point counts
are trajectory-level metadata because object identity and reference vertices
must remain stable across a sampled T=8 window. Isaac Sim exporters must split
a simulation when objects are spawned/despawned or a mesh topology changes.

For large datasets, `--train-data` may instead point to a directory containing
memory-mapped `positions.npy`, `props.npy`, `object_lens.npy`, and
`point_lens.npy` arrays with the same shapes. The dataset trims archive-level
padding before augmentation, and the dedicated collate function dynamically
zero-pads only to the current batch maxima. Both length tensors are passed
through the DDP entry into the model's combined object/point mask. Use
`--require-length-metadata` for Isaac Sim or any padded archive so accidentally
omitted masks fail immediately. Legacy archives containing only `positions`
and `props` remain supported and are interpreted as fully valid, fixed-size
data.

For legacy `.npz` or `.npy` data, `--base-physical-dt` remains required and
means the interval between consecutive saved frames. For HDF5 it is optional;
if supplied, it must equal the file's `record_dt_s`, so accidentally passing
the smaller PhysX internal step fails immediately.

Checkpoint `latest.pt` is written atomically every five epochs and at completion. It
contains model, optimizer, scheduler, AMP scaler, global step, and per-rank RNG
states. `--resume auto` resumes it at the next epoch and requires the same DDP
world size for exact reproducibility. Plain FP32 is the default because the
paper does not disclose mixed precision; `--amp bf16` and `--amp fp16` are
explicit performance options.

For variable-object batches, the DDP objective is normalized over valid objects
across the complete global batch, not independently on each rank. Each local
loss is scaled by `world_size * local_valid_objects / global_valid_objects`
before backward so DDP's gradient average is exactly equivalent to one global
masked mean. Step and epoch metrics use the same valid-object weighting. The
checkpoint records this loss-reduction protocol and intentionally refuses to
resume older checkpoints that would change the optimization objective mid-run.

## Box2D example

Train on a synthetic Box2D dataset of boxes sliding on the ground under friction and coming to rest, and save `examples/box2d.png` + `examples/box2d_trajectories.png` (training loss, a held-out rollout vs ground truth, and rollout error against the constant-velocity baseline):

```bash
$ uv run --extra examples python examples/train_box2d.py --example friction    # smooth dynamics
$ uv run --extra examples python examples/train_box2d.py --example collision   # sharp contacts
```

## Citations

```bibtex
@misc{dou2026rigidformerlearningrigiddynamics,
    title   = {RigidFormer: Learning Rigid Dynamics using Transformers},
    author  = {Zhiyang Dou and Minghao Guo and Haixu Wu and Doug Roble and Tuur Stuyck and Wojciech Matusik},
    year    = {2026},
    eprint  = {2605.09196},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2605.09196},
}
```

```bibtex
@inproceedings{Arora2023ZoologyMA,
    title   = {Zoology: Measuring and Improving Recall in Efficient Language Models},
    author  = {Simran Arora and Sabri Eyuboglu and Aman Timalsina and Isys Johnson and Michael Poli and James Zou and Atri Rudra and Christopher R'e},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:266149332}
}
```

```bibtex
@misc{islam2026platonictransformerssolidchoice,
    title   = {Platonic Transformers: A Solid Choice For Equivariance}, 
    author  = {Mohammad Mohaiminul Islam and Rishabh Anand and David R. Wessels and Friso de Kruiff and Thijs P. Kuipers and Rex Ying and Clara I. Sánchez and Sharvaree Vadgama and Georg Bökman and Erik J. Bekkers},
    year    = {2026},
    eprint  = {2510.03511},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2510.03511}, 
}
```
