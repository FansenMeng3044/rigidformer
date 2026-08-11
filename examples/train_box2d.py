"""Train Rigidformer on a toy Box2D dataset and plot the result.

    uv run --extra examples python examples/train_box2d.py --example friction --steps 1000

    friction   boxes slide under friction and stop - smooth dynamics, the model's strength
    collision  boxes fall, bounce, and collide - sharp contacts, the model's weakness
"""

import fire

import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import default_collate

from box2d_dataset import Box2DDataset, trajectory, NUM_OBJECTS, STRIDE
from rigidformer import Rigidformer, RigidformerRolloutWrapper

DT = STRIDE / 60.

def main(
    example = 'friction',
    steps = 1000,
    num_train = 64,
    eval_every = 500,
    seed = 42
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    collide = example == 'collision'

    # data - disjoint train / val trajectories

    train = Box2DDataset([trajectory(seed, collide) for seed in range(num_train)], collide)
    val = Box2DDataset([trajectory(seed, collide) for seed in range(1000, 1016)], collide)

    # model

    model = Rigidformer(
        dim = 128,
        dim_head = 32,
        heads = 4,
        num_register_tokens = 4,
        object_self_attn_depth = 3,
        anchor_cross_attn_depth = 3,
        object_hidden_layers = (0, 1, 3),
        num_anchors = 4,
        vertex_properties_dim = 3
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, weight_decay = 1e-2)

    # train

    losses = []
    eval_curve = []

    # rollout baseline - constant velocity extrapolation

    def const_vel_rollout(gt):
        prev, cur = gt[0], gt[1]
        const_vel = [prev, cur]
        for _ in range(11):
            const_vel.append(2 * const_vel[-1] - const_vel[-2])
        return torch.stack(const_vel)

    def center(frames):
        return frames.mean(dim = -2)

    def rollout_error():
        model.eval()

        positions, props = val.trajectories[0]
        start_frame = 30 if not collide else 10
        gt = torch.from_numpy(positions[start_frame:start_frame + STRIDE * 13:STRIDE])[:13].unsqueeze(1)

        wrapper = RigidformerRolloutWrapper(model)

        with torch.no_grad():
            preds = torch.stack(wrapper(
                delta_times = torch.full((1,), DT),
                vertex_properties = torch.from_numpy(props).unsqueeze(0),
                object_positions = [gt[:1, 0], gt[1:2, 0]],
                num_steps = 11
            ))

        model_err = (center(preds) - center(gt)).norm(dim = -1)[:, 0].mean(dim = -1)[-1].item()
        const_vel_err = (center(const_vel_rollout(gt)) - center(gt)).norm(dim = -1)[:, 0].mean(dim = -1)[-1].item()

        model.train()
        return model_err, const_vel_err

    for step in range(steps):
        batch = default_collate([train[np.random.randint(len(train))] for _ in range(8)])

        loss, _ = model(**batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % eval_every == 0:
            model_err, const_vel_err = rollout_error()
            eval_curve.append((step + 1, model_err, const_vel_err))
            print(f'step {step + 1:5d} | loss {loss.item():.4f} | rollout {model_err:.4f} m | const-vel {const_vel_err:.4f} m')

    # evaluate - single-step rmse vs baselines

    model.eval()

    rmse = dict(model = [], const_vel = [], stay = [])

    with torch.no_grad():
        for _ in range(32):
            batch = default_collate([val[np.random.randint(len(val))]])
            gt_next = batch.pop('object_pos_next')

            pred = model(**batch).object_pos_next

            for key, p in (('model', pred), ('const_vel', 2 * batch['object_pos'] - batch['object_pos_prev']), ('stay', batch['object_pos'])):
                rmse[key].append((p - gt_next).norm(dim = -1).mean().item())

    rmse = {key: np.mean(value) for key, value in rmse.items()}

    print('single-step rmse:', {key: round(value, 4) for key, value in rmse.items()})

    # evaluate - rollout on a held-out trajectory

    positions, props = val.trajectories[0]
    start_frame = 30 if not collide else 10
    gt = torch.from_numpy(positions[start_frame:start_frame + STRIDE * 13:STRIDE])[:13].unsqueeze(1)

    wrapper = RigidformerRolloutWrapper(model)

    with torch.no_grad():
        preds = torch.stack(wrapper(
            delta_times = torch.full((1,), DT),
            vertex_properties = torch.from_numpy(props).unsqueeze(0),
            object_positions = [gt[:1, 0], gt[1:2, 0]],
            num_steps = 11
        ))

    model_err = (center(preds) - center(gt)).norm(dim = -1)[:, 0].mean(dim = -1)
    const_vel_err = (center(const_vel_rollout(gt)) - center(gt)).norm(dim = -1)[:, 0].mean(dim = -1)

    print(f'rollout error @12 steps: model {model_err[-1].item():.4f} m | const-vel {const_vel_err[-1].item():.4f} m')

    # plots

    ema = [losses[0]]
    for value in losses[1:]:
        ema.append(.05 * value + .95 * ema[-1])

    gt_c, pred_c = center(gt)[:, 0], center(preds)[:, 0]

    def frame_df(coords, source):
        return pd.DataFrame(dict(
            time = np.tile(np.arange(13) * DT, NUM_OBJECTS),
            obj = np.repeat(np.arange(NUM_OBJECTS), 13),
            x = coords[..., 0].numpy().ravel(),
            z = coords[..., 2].numpy().ravel(),
            source = source
        ))

    traj = pd.concat((frame_df(gt_c, 'gt'), frame_df(pred_c, 'rollout')))

    loss_df = pd.DataFrame(dict(
        step = np.tile(np.arange(len(losses)), 2),
        loss = np.concatenate((losses, ema)),
        source = np.repeat(('raw', 'ema'), len(losses))
    ))

    err_df = pd.DataFrame(dict(
        step = np.tile(np.arange(13), 2),
        rmse = np.concatenate((model_err, const_vel_err)),
        model = np.repeat(('rigidformer', 'const-vel baseline'), 13)
    ))

    curve_df = pd.DataFrame(eval_curve, columns = ('step', 'rigidformer', 'const-vel baseline'))
    curve_df = curve_df.melt('step', var_name = 'model', value_name = 'rmse')

    fig, axes = plt.subplots(2, 2, figsize = (15, 9))

    sns.lineplot(data = loss_df, x = 'step', y = 'loss', hue = 'source', ax = axes[0, 0])
    axes[0, 0].set_title('training loss')

    coord = 'x' if not collide else 'z'
    sns.lineplot(data = traj, x = 'time', y = coord, hue = 'obj', style = 'source', ax = axes[0, 1])
    axes[0, 1].set_title(f'box {coord}-position: gt (solid) vs rollout (dashed)')
    axes[0, 1].set_ylabel(f'{coord} (m)')

    sns.lineplot(data = err_df, x = 'step', y = 'rmse', hue = 'model', ax = axes[1, 0])
    axes[1, 0].set_title('rollout error vs horizon')

    sns.lineplot(data = curve_df, x = 'step', y = 'rmse', hue = 'model', ax = axes[1, 1])
    axes[1, 1].set_title('rollout error vs training steps')

    fig.tight_layout()
    fig.savefig(f'examples/box2d_{example}.png', dpi = 150)

    # top-down view of the arena - box outlines fading from light (t0) to dark (tlast)

    frames = (0, 3, 6, 9, 12)
    time_labels = [f't={frame * DT:.1f}s' for frame in frames]

    fig2, axes2 = plt.subplots(1, 2, figsize = (13, 5.5))

    for ax, (coords, title) in zip(axes2, ((gt[:, 0], 'ground truth'), (preds[:, 0], 'predicted'))):
        for obj in range(NUM_OBJECTS):
            for i, frame in enumerate(frames):
                pts = coords[frame, obj][:, [0, 2]].numpy()
                pts = np.concatenate((pts, pts[:1]))  # close the rectangle
                ax.plot(pts[:, 0], pts[:, 1], color = f'C{obj}', alpha = .25 + .75 * i / (len(frames) - 1), lw = 1.2)

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-.2, 5)

    handles = [plt.Line2D([], [], color = f'C{o}', lw = 2, label = f'object {o}') for o in range(NUM_OBJECTS)]
    handles += [plt.Line2D([], [], color = 'k', alpha = .25 + .75 * i / (len(frames) - 1), lw = 2, label = label) for i, label in enumerate(time_labels)]
    axes2[0].legend(handles = handles, fontsize = 8, loc = 'upper right')

    fig2.tight_layout()
    fig2.savefig(f'examples/box2d_{example}_trajectories.png', dpi = 150)

    print(f'saved examples/box2d_{example}.png and examples/box2d_{example}_trajectories.png')

if __name__ == '__main__':
    fire.Fire(main)
