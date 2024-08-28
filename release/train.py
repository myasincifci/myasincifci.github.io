import argparse
from pathlib import Path
from typing import List, Optional
import time 
import math
import os
import copy
import random
import torch
import matplotlib.pyplot as plt
from renderer import Renderer
from tqdm import tqdm
from utils import readCamerasFromTransforms, CameraInfo
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

profiler = False

def image_path_to_tensor(image_path):
    img = Image.open(image_path)
    img_tensor = TF.to_tensor(img)[:3]
    return img_tensor

def init_parameters(N, device):
    # Random gaussians
    bd = 2.3
    be = 0.02

    mu = bd * (torch.rand(N, 3, device=device,) - 0.5)
    scales = be * (torch.ones(N, 3, device=device,))
    d = 3
    cols = torch.ones(N, d, device=device,) * 0.5

    u = torch.rand(N, 1, device=device)
    v = torch.rand(N, 1, device=device)
    w = torch.rand(N, 1, device=device)

    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    quats = quats.to(device=device)
    opcs = torch.rand((N,), device=device,)

    return {
        'mu': mu, 'scales': scales, 'quats': quats, 'cols': cols, 'opcs': opcs
    }

def init_scene(root: str):
    scene = readCamerasFromTransforms(
        root,
        'transforms_train.json',
        False
    )

    return scene

def get_view(scene: List[CameraInfo], W, H, queue: List):
    if not queue:
        queue = copy.copy(scene)
        random.shuffle(queue)

    view = queue.pop()

    gt_image = TF.resize(image_path_to_tensor(view.image_path), (H, W)).permute(1,2,0)
    viewmat = torch.from_numpy(view.w2c).to(dtype=torch.float)
    focal = 0.5 * float(W) / math.tan(0.5 * view.FovX)
    camera = {'gt_image': gt_image, 'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}

    return camera

def get_test_view(root, index, W, H):
    scene = readCamerasFromTransforms(
        root,
        'transforms_test.json',
        False
    )

    view = scene[index]

    gt_image = TF.resize(image_path_to_tensor(view.image_path), (H, W)).permute(1,2,0)
    viewmat = torch.from_numpy(view.w2c).to(dtype=torch.float)
    focal = 0.5 * float(W) / math.tan(0.5 * view.FovX)
    camera = {'gt_image': gt_image, 'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}

    return camera

def main(args: argparse.Namespace):
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 10_000
    W = H = 256

    params = init_parameters(N, device)   

    renderer = Renderer(params=params, device=device, tile_size=args.tile_size)

    # Load scene
    scene = init_scene(root=f'./nerf_synthetic/{args.dataset}')

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.parameters(), lr=1e-3)

    test_view = get_test_view(root=f'./nerf_synthetic/{args.dataset}', index=0, W=W, H=H)

    plt.ion()
    figure, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.matshow(torch.rand(H, W))
    ax1.set_title('Test View')
    im2 = ax2.matshow(test_view['gt_image'])
    ax2.set_title('Test GT')

    frames = []
    camera_queue = []
    for iter in tqdm(range(5_000)):

        view = get_view(scene, W, H, camera_queue)

        optimizer.zero_grad()

        forward_start = time.time()
        pred = renderer(view)
        forward_end = time.time()

        pred[pred.isnan() | pred.isinf()] = 0.

        with torch.no_grad():
            pred_test = renderer(test_view)

            im1.set_data(pred_test.detach().cpu())

            if iter % 10 == 0:
                frames.append((pred_test.detach().cpu().numpy() * 255).astype(np.uint8))

        figure.canvas.draw()
        figure.canvas.flush_events()

        loss = criterion(pred, view['gt_image'])
        
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()

        # Filter inf and nan values 
        for param in renderer.parameters():
            param.grad[param.grad.isnan()] = 0.
            param.grad[param.grad.isinf()] = 0.

        optimizer.step()

        print(f'Iter: {iter}, Fwd.: {(forward_end - forward_start):.3f}s, Bckwd.: {(backward_end - backward_start):.3f}s')
        print(f'Loss: {loss.item()}, Grad. Norms: {[p.abs().norm().item() for p in renderer.parameters()]}')


    # save them as a gif with PIL
    frames = [Image.fromarray(frame) for frame in frames]
    out_dir = os.path.join(os.getcwd(), "renders")
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{args.dataset}"
    Path(path).mkdir(parents=True, exist_ok=True)
    frames[-1].save(
        os.path.join(path, "final.png")
    )
    gt = Image.fromarray(view['gt_image'])
    gt.save(
        os.path.join(path, "ground_truth.png")
    )
    frames[0].save(
        os.path.join(path, f"training-{N}.gif"),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=5,
        loop=0,)

    plt.matshow(pred.detach().cpu())
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile-size', type=int, dest='tile_size', default=32)
    parser.add_argument('--dataset', type=str, dest='dataset', 
                        default='drums', choices=['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'])

    args = parser.parse_args()

    main(args)