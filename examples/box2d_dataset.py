"""Tiny Box2D rigid-body dataset, in two flavors.

    friction   boxes slide on the ground under friction and come to rest
               (no impacts - smooth, easy to learn)

    collision  boxes fall, bounce off the ground, and collide with each
               other (sharp contacts - hard to learn)

Box2D (x, y) maps to 3D (x, z); the ground is z = 0. Points are sampled on
the rectangle perimeter (plus a small out-of-plane thickness) and tracked
rigidly, so vertex correspondence is exact across frames.
"""

import numpy as np
import torch
import fire
from torch.utils.data import Dataset

from Box2D import b2FixtureDef
from Box2D.b2 import world, polygonShape, edgeShape

# constants

GRAVITY = (0., -3.)
PHYS_DT = 1. / 60.          # physics timestep (s)
NUM_FRAMES = 361            # physics frames per trajectory (6 s)
NUM_OBJECTS = 3
NUM_POINTS = 48
THICKNESS = .05
STRIDE = 10                 # one training step = 10 physics frames (1/6 s)

# helper

def points_for_box(hx, hz, n = NUM_POINTS, thickness = THICKNESS):
    """perimeter points of a box in (x, z), duplicated at +-thickness in y"""

    corners = [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]

    points = []
    per_edge = n // 8

    for (a, b) in edges:
        for k in range(per_edge):
            s = (k + .5) / per_edge
            x = a[0] + s * (b[0] - a[0])
            z = a[1] + s * (b[1] - a[1])
            points += [(x, -thickness, z), (x, thickness, z)]

    return np.array(points, dtype = np.float32)

def trajectory(seed, collide = False):
    """one rollout of boxes in an arena, positions tracked rigidly frame by frame"""

    rng = np.random.default_rng(seed)

    w = world(gravity = GRAVITY)

    # arena - ground plane plus side walls

    w.CreateStaticBody(position = (0, -.5), shapes = polygonShape(box = (11, .5)))
    for x in (-10., 10.):
        w.CreateStaticBody(shapes = edgeShape(vertices = ((x, 0), (x, 20))))

    # boxes - dropped with spin (collision) or sliding (friction)

    bodies, points, props = [], [], []

    for _ in range(NUM_OBJECTS):
        hx, hz = rng.uniform(.3, 1.2, size = 2)

        if collide:
            body = w.CreateDynamicBody(
                position = (rng.uniform(-6, 6), rng.uniform(1.5, 4)),
                angle = rng.uniform(0, 2 * np.pi),
                linearVelocity = (rng.uniform(-2., 2.), 0.),
                angularVelocity = rng.uniform(-1.5, 1.5)
            )

            restitution = rng.uniform(.5, .7)
            group_index = 0  # collide with each other
        else:
            body = w.CreateDynamicBody(
                position = (rng.uniform(-6, 6), hz + .01),
                linearVelocity = (rng.uniform(1.5, 3.5), 0.)
            )

            restitution = 0.
            group_index = -1  # pass through each other

        fixture_def = b2FixtureDef(
            shape = polygonShape(box = (hx, hz)),
            density = rng.uniform(.5, 2.),
            friction = rng.uniform(.3, .5),
            restitution = restitution
        )

        fixture_def.filter.groupIndex = group_index
        body.CreateFixture(fixture_def)

        bodies.append(body)
        points.append(points_for_box(hx, hz))
        props.append((body.mass, 4 * hx * hz, hz))  # mass, area, half-height

    # track points rigidly - correspondence exact across frames

    positions = np.zeros((NUM_FRAMES, NUM_OBJECTS, NUM_POINTS, 3), dtype = np.float32)

    for frame in range(NUM_FRAMES):
        for i, (body, pts) in enumerate(zip(bodies, points)):
            c, s = np.cos(body.angle), np.sin(body.angle)
            positions[frame, i, :, 0] = body.position.x + c * pts[:, 0] - s * pts[:, 2]
            positions[frame, i, :, 1] = pts[:, 1]
            positions[frame, i, :, 2] = body.position.y + s * pts[:, 0] + c * pts[:, 2]

        w.Step(PHYS_DT, 6, 2)

    return positions, np.array(props, dtype = np.float32)

class Box2DDataset(Dataset):
    """windows of (prev, cur, next) frames

    half the windows are oversampled from the hardest event to learn:
    the stop transition for friction, the hard-contact instant for collision
    """

    def __init__(self, trajectories, collide = False):
        self.trajectories = trajectories
        self.collide = collide

    def __len__(self):
        return len(self.trajectories) * 16

    def __getitem__(self, index):
        positions, props = self.trajectories[index % len(self.trajectories)]

        # oversample the hard events

        max_start = NUM_FRAMES - 2 * STRIDE - 1
        start = int(np.random.randint(0, max_start))

        for _ in range(64):
            if np.random.rand() < .5:
                break

            if self.collide:
                accel = np.abs(positions[start + 2 * STRIDE] - 2 * positions[start + STRIDE] + positions[start]).max() / STRIDE ** 2
                if accel > .02:
                    break
            else:
                velocity = positions[start + STRIDE] - positions[start]
                prev_velocity = positions[start] - positions[max(0, start - STRIDE)]

                stopping = ((np.abs(velocity[..., 0]).max() < .15) & (np.abs(prev_velocity[..., 0]).max() > .3)).any()

                if stopping:
                    break

            start = int(np.random.randint(0, max_start))

        return dict(
            object_pos_prev = torch.from_numpy(positions[start]),
            object_pos = torch.from_numpy(positions[start + STRIDE]),
            object_pos_next = torch.from_numpy(positions[start + 2 * STRIDE]),
            object_first_frame_pos = torch.from_numpy(positions[start]),
            vertex_properties = torch.from_numpy(props),
            delta_times = torch.tensor(STRIDE / 60.)  # seconds
        )

# cli

def generate(
    seed = 0,
    collide = False,
    num_trajectories = 16,
    path = 'box2d_trajectories.npz'
):
    """roll out trajectories and save them to a .npz archive for offline use"""

    positions, props = [], []

    for i in range(num_trajectories):
        p, pr = trajectory(seed + i, collide)
        positions.append(p)
        props.append(pr)

    positions = np.stack(positions)
    props = np.stack(props)

    np.savez(path, positions = positions, props = props)
    print(f'saved {num_trajectories} trajectories to {path}')

if __name__ == '__main__':
    fire.Fire(dict(
        generate = generate,
        trajectory = trajectory,
        dataset = Box2DDataset
    ))
