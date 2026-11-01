import json
import yaml
import torch
from scipy.ndimage import measurements
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset, Kernel_Dataset
from sampler import get_sampler, Trajectory
from model import get_model
from eval import get_eval_fn, Evaluator
from torch.nn.functional import interpolate
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb
import setproctitle
from PIL import Image
import numpy as np
import imageio
import os

import cv2

def resize_to_img(x, target_hw):
    """Resize H×W×3 image to target size"""
    th, tw = target_hw
    return cv2.resize(x, (tw, th), interpolation=cv2.INTER_NEAREST)


def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        if x_.shape[0] == 3:
            np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        else:
            np_x = norm(x_[0]).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)
def traj_to_np(traj):
    """
    traj:
        [T,B,C,H,W]  (image)
        [T,B,H,W]    (kernel / grayscale)
    return:
        [T,B,H,W,3]  uint8
    """
    T, B = traj.shape[:2]

    # ---------- Case 1: 有通道 ----------
    if traj.ndim == 5:
        traj = traj.reshape(T * B, *traj.shape[2:])  # [TB,C,H,W]
        np_traj = tensor_to_numpy(traj)              # [TB,H,W,C]

    # ---------- Case 2: 无通道 (k变量) ----------
    elif traj.ndim == 4:
        traj = traj.reshape(T * B, 1, *traj.shape[2:])  # 加 channel 维
        np_traj = tensor_to_numpy(traj)                 # [TB,H,W,1]
        np_traj = np.repeat(np_traj, 3, axis=-1)        # 灰度 → RGB

    else:
        raise ValueError(f"Unsupported traj shape {traj.shape}")

    np_traj = np_traj.reshape(T, B, *np_traj.shape[1:])  # [T,B,H,W,C]
    return np_traj

def save_mp4_video(
        gt, y,
        x0hat_traj, x0y_traj, xt_traj,
        output_path,
        fps=24, sec=5, space=4,
        idx=0
):
    """
    Visualize diffusion trajectory.
    All traj shape: [T,B,C,H,W] or [T,B,H,W]
    """

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)

    # ===== 时间重采样 =====
    T = len(xt_traj)
    reindex = np.linspace(0, T - 1, sec * fps).astype(int)

    # ===== 转 numpy =====
    np_x0hat_traj = traj_to_np(x0hat_traj[reindex])
    np_x0y_traj   = traj_to_np(x0y_traj[reindex]) if x0y_traj is not None else None
    np_xt_traj    = traj_to_np(xt_traj[reindex])

    sample = np_x0hat_traj[0, idx]
    H, W = sample.shape[:2]

    np_y  = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]

    if np_y.shape[:2] != (H, W):
        np_y = resize_to_img(np_y,(H, W))
        np_gt = resize_to_img(np_gt,(H, W))

    width = 5 * W + 4 * space
    height = H

    # ===== 逐帧写入 =====
    for t in range(len(reindex)):

        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        x0hat = np_x0hat_traj[t, idx]
        xt    = np_xt_traj[t, idx]
        x0y   = np_x0y_traj[t, idx] if np_x0y_traj is not None else np_y

        cx, cy = 0, 0

        canvas[cx:cx+H, cy:cy+W] = np_y
        cy += W + space

        canvas[cx:cx+H, cy:cy+W] = np_gt
        cy += W + space

        canvas[cx:cx+H, cy:cy+W] = x0y
        cy += W + space

        canvas[cx:cx+H, cy:cy+W] = x0hat
        cy += W + space

        canvas[cx:cx+H, cy:cy+W] = xt

        writer.append_data(canvas)

    writer.close()



def log_results(args, sde_trajs, results, images, y, full_samples, kernel_samples, kernel_gts, table_markdown, total_number):
    # log hyperparameters and configurations
    full_samples = full_samples.flatten(0, 1)
    kernel_samples = kernel_samples.flatten(0, 1)
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)

    # log grid results
    resized_y = resize(y, images, args.task[args.task_group].operator.name)
    stack = torch.cat([images, resized_y, full_samples])
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=total_number)

    # log individual sample instances
    if args.save_samples:
        pil_image_list = tensor_to_pils(full_samples)
        image_dir = safe_dir(root / 'samples')
        cnt = 0
        for run in range(args.num_runs):
            for idx in range(total_number):
                image_path = image_dir / '{:05d}_run{:04d}.png'.format(idx, run)
                pil_image_list[cnt].save(str(image_path))
                cnt += 1

    # # log sampling trajectory and mp4 video
    # if args.save_traj:
    #     traj_dir = safe_dir(root / 'trajectory')
    #     print()
    #     print('save trajectories to mp4 videos...')
    #     for run, sde_traj in enumerate(sde_trajs):
    #         if args.save_traj_raw_data:
    #             # might be SUPER LARGE
    #             traj_raw_data = safe_dir(traj_dir / 'raw')
    #             torch.save(sde_traj, str(traj_raw_data / 'trajectory_run{:04d}.pth'.format(run)))
    #         # save mp4 video for trajectories
    #         x0hat_traj = sde_traj.tensor_data['x0hat']
    #         x0y_traj = sde_traj.tensor_data['x0y']
    #         xt_traj = sde_traj.tensor_data['xt']
    #         for idx in range(total_number):
    #             video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx, run))
    #             save_mp4_video(images[idx], resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)
    # log sampling trajectory and mp4 video
    if args.save_traj:
        traj_dir = safe_dir(root / 'trajectory')
        print('\nsave trajectories to mp4 videos...')

        for run, sde_traj_dict in enumerate(sde_trajs):

            if args.save_traj_raw_data:
                traj_raw_data = safe_dir(traj_dir / 'raw')
                for model_name, traj in sde_traj_dict.items():
                    torch.save(
                        traj,
                        str(traj_raw_data / f'trajectory_{model_name}_run{run:04d}.pth')
                    )

            for model_name, traj in sde_traj_dict.items():

                td = traj.tensor_data
                x0hat_traj = td.get('x0hat')
                x0y_traj = td.get('x0y')
                xt_traj = td.get('xt')

                if x0hat_traj is None or xt_traj is None:
                    print(f'[Warning] {model_name} missing trajectory.')
                    continue

                for idx in range(total_number):
                    video_path = str(
                        traj_dir / f'{idx:05d}_{model_name}_run{run:04d}.mp4'
                    )

                    save_mp4_video(
                        gt=images[idx],
                        y=resized_y[idx],
                        x0hat_traj=x0hat_traj,
                        x0y_traj=x0y_traj,
                        xt_traj=xt_traj,
                        output_path=video_path,
                        idx=idx
                    )
    # log the evaluation metrics
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(table_markdown)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)

    # log kernel samples (optional)
    if args.save_samples:
        kernel_pil_list = tensor_to_pils(kernel_samples)
        kernel_dir = safe_dir(root / 'kernel_samples')
        cnt = 0
        for run in range(args.num_runs):
            for idx in range(total_number):
                kernel_path = kernel_dir / '{:05d}_run{:04d}.png'.format(idx, run)
                kernel_pil_list[cnt].save(str(kernel_path))
                cnt += 1
    # log the kernel_gt and measurment (optional)
    if args.save_samples:
        if args.sampler['blind']:
            kernel_gts_pil_list = tensor_to_pils(kernel_gts)
            kernel_dir = safe_dir(root / 'kernel_gts')
            cnt = 0
            for idx in range(total_number):
                kernel_gt_path = kernel_dir / '{:05d}.png'.format(idx)
                kernel_gts_pil_list[cnt].save(str(kernel_gt_path))
                cnt += 1

        measurements_pil_list = tensor_to_pils(y)
        measurements_dir = safe_dir(root / 'measurements')
        cnt = 0
        for idx in range(total_number):
            measurements_gt_path = measurements_dir / '{:05d}.png'.format(idx)
            measurements_pil_list[cnt].save(str(measurements_gt_path))
            cnt += 1

    # log the evaluation metrics
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(table_markdown)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)


# def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, record, batch_size, gt):
#     """
#         posterior sampling in batch
#     """
#     samples = []
#     trajs = []
#     for s in range(0, len(x_start), batch_size):
#         # update evaluator to correct batch index
#         cur_x_start = x_start[s:s + batch_size]
#         cur_y = y[s:s + batch_size]
#         cur_gt = gt[s: s + batch_size]
#         cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, evaluator, verbose=verbose, record=record, gt=cur_gt)
#
#         samples.append(cur_samples)
#         if record:
#             trajs.append(sampler.trajectory.compile())
#     if record:
#         trajs = Trajectory.merge(trajs)
#     return torch.cat(samples, dim=0), trajs

# Define blind deblur function
def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, record, batch_size, gt):
    """
        posterior sampling in batch
        Modified to handle a dictionary input/output for 'image' and 'kernel'.
    """
    samples = []  # Store combined samples
    trajs = []   # Store combined trajectories

    # Loop over batches
    for s in range(0, len(x_start['image']), batch_size):  # Use the length of 'image' key
        # Extract current batch for both 'image' and 'kernel'
        cur_x_start = {
            'image': x_start['image'][s:s + batch_size],
            'kernel': x_start['kernel'][s:s + batch_size]
        }
        cur_y = y[s:s + batch_size]  # Measurement remains the same
        cur_gt = {
            'image': gt['image'][s:s + batch_size],
            'kernel': gt['kernel'][s:s + batch_size]
        }

        # Perform sampling with the entire dictionary as input
        cur_samples = sampler.sample(
            model_dict=model,                  # Entire model dictionary
            x_start=cur_x_start,         # Entire x_start dictionary
            operator=operator,           # Operator is shared
            measurement=cur_y,                     # Measurement is shared
            evaluator=evaluator,         # Evaluator is shared
            verbose=verbose,
            record=record,
            gt=cur_gt                    # Entire gt dictionary
        )

        # Append results
        samples.append(cur_samples)
        if record:
            if isinstance(sampler.trajectory, dict):
                trajs.append({
                    k: v.compile() for k, v in sampler.trajectory.items()
                })
            else:
                trajs.append(sampler.trajectory.compile())
    # Concatenate all batches
    samples = {key: torch.cat([batch[key] for batch in samples], dim=0) for key in samples[0].keys()}
    if record:
        trajs = Trajectory.merge(trajs)

    return samples, trajs


@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')
def main(args):
    # Fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.gpu))

    setproctitle.setproctitle(args.name)
    print(args)

    # Get data
    data = get_dataset(**args.data)
    total_number = len(data)
    images = data.get_data(total_number, 0)

    # Get kernel
    kernel_dataset = Kernel_Dataset(**args.kernel_data)  # Define kernel_dataset class according to task type
    kernels = kernel_dataset.get_kernel(total_number)       # Get the same number of kernels as the images

    # Get operator & measurement
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    packed_data = {'image': images, 'kernel': kernels}
    # y = images     # test for deblurring real images
    y = operator.measure(packed_data)  # Blind Deblur
    # y = operator.measure(images)     # Non-Blind Deblur

    # Get sampler
    sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.mcmc_sampler_config)

    # Get model
    image_model = get_model(**args.model)
    model = (
        {'image': image_model, 'kernel': get_model(**args.kernel_model)}
        if hasattr(args, 'kernel_model')
        else image_model
    )

    # --- Create separate evaluators for reporting ---
    # image evaluator: use image metrics specified (e.g., psnr, ssim, lpips)
    image_eval_fns = []
    for name in args.image_eval_fn_list if hasattr(args, 'image_eval_fn_list') else args.eval_fn_list:
        image_eval_fns.append(get_eval_fn(name))
    evaluator_image = Evaluator(tuple(image_eval_fns))

    # kernel evaluator: typically only kernel_mse
    kernel_eval_fns = []
    # allow args.kernel_eval_fn_list override, else 'kernel_mse'
    kernel_names = args.kernel_eval_fn_list if hasattr(args, 'kernel_eval_fn_list') else ('kernel_mse',)
    for name in kernel_names:
        kernel_eval_fns.append(get_eval_fn(name))
    evaluator_kernel = Evaluator(tuple(kernel_eval_fns))

    # +++++++++++++++++++++++++++++++++++
    # Main sampling process
    full_samples = {'image': [], 'kernel': []} # Store samples for both image and kernel
    full_trajs = []                             # Store trajectories for all runs

    for r in range(args.num_runs):
        print(f'Run: {r}')
        x_start = sampler.get_start(images.shape[0], model)
        samples, trajs = sample_in_batch(
            sampler=sampler,
            model=model,
            x_start=x_start,
            operator=operator,
            y=y,
            evaluator=evaluator_image,
            verbose=True,
            record=args.save_traj,
            batch_size=args.batch_size,
            gt=packed_data
        )
        # Append results for this run
        for key in ['image', 'kernel']:
            full_samples[key].append(samples[key])  # Append samples for each key
        if args.save_traj:
            full_trajs.append(trajs)               # Append trajectories

    # Concatenate all runs into a single tensor
    full_samples = {
        key: torch.stack(full_samples[key], dim=0) for key in full_samples.keys()
    }

    # Log metrics
    # Log metrics for 'image'
    results_image = evaluator_image.report(packed_data['image'], y, full_samples['image'])
    markdown_text_image = evaluator_image.display(results_image)
    print(markdown_text_image)

    if args.sampler['blind']:
        # Log metrics for 'kernel'
        norm_k = (full_samples['kernel'] * 0.5 + 0.5).clip(0, 1)
        norm_k = norm_k / norm_k.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
    else:
        norm_k = full_samples['kernel']

    results_kernel = evaluator_kernel.report(packed_data['kernel'], y, norm_k)
    markdown_text_kernel = evaluator_kernel.display(results_kernel)
    print(markdown_text_kernel)

    # Combine the results and markdown texts
    results = {
        'image': results_image,
        'kernel': results_kernel
    }

    # Log results
    log_results(
        args=args,
        sde_trajs=full_trajs,
        results=results,
        images=packed_data['image'],
        y=y,
        full_samples=full_samples['image'],
        kernel_samples=full_samples['kernel'],
        kernel_gts=kernels/kernels.max() * 2 - 1,
        table_markdown=("### Image Evaluation\n" + markdown_text_image +
    "\n\n### Kernel Evaluation\n" + markdown_text_kernel),
        total_number=total_number
    )
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=args.name,
            config=OmegaConf.to_container(args, resolve=True)
        )
        evaluator_image.log_wandb(results, args.batch_size)
    print(f'Finish {args.name}!')
# def main(args):
#     # fix random seed
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.cuda.set_device('cuda:{}'.format(args.gpu))
#
#     setproctitle.setproctitle(args.name)
#     print(args)
#
#     # get data
#     data = get_dataset(**args.data)
#     total_number = len(data)
#     images = data.get_data(total_number, 0)
#
#     # get kernel
#     task_group = args.task[args.task_group]
#     kernel_dataset = Kernel_Dataset( **task_group.operater)    # Define kernel_dataset class according to task type
#     kernels = kernel_dataset.get_kernel(total_number)          # Get the same number of kernels as the images
#
#     # get operator & measurement
#     # Todo: Need to integrate blind and non-blind methods
#     operator = get_operator(**task_group.operator)
#     packed_data = {'image': images, 'kernel': kernels}
#     y = operator.measure(packed_data)   # Blind Deblur
#     # y = operator.measure(images)    # Non-Blind Deblur
#
#     # get sampler
#     sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.mcmc_sampler_config)
#
#     # get model
#     model = {
#         'image': get_model(**args.model),
#         'kernel': get_model(**args.kernel_model) if hasattr(args, 'kernel_model') else None
#     }
#     # model = get_model(**args.model)    # Non-Blind Deblur
#
#     # get evaluator
#     eval_fn_list = []
#     for eval_fn_name in args.eval_fn_list:
#         eval_fn_list.append(get_eval_fn(eval_fn_name))
#     evaluator = Evaluator(eval_fn_list)
#
#     # +++++++++++++++++++++++++++++++++++
#     # main sampling process
#     full_samples = []
#     full_trajs = []
#     for r in range(args.num_runs):
#         print(f'Run: {r}')
#         x_start = {'image':sampler.get_start(images.shape[0], model['image']),
#                    'kernel':sampler.get_start(kernels.shape[0], model['kernel'])}
#         samples, trajs = sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose=True,
#                                          record=args.save_traj, batch_size=args.batch_size, gt=packed_data)
#         full_samples.append(samples)
#         full_trajs.append(trajs)
#     full_samples = torch.stack(full_samples, dim=0)
#
#     # log metrics
#     results = evaluator.report(images, y, full_samples)
#     markdown_text = evaluator.display(results)
#     print(markdown_text)
#
#     # log results
#     log_results(args, full_trajs, results, images, y, full_samples, markdown_text, total_number)
#     if args.wandb:
#         wandb.init(
#             project=args.project_name,
#             name=args.name,
#             config=OmegaConf.to_container(args, resolve=True)
#         )
#         evaluator.log_wandb(results, args.batch_size)
#     print(f'finish {args.name}!')


if __name__ == '__main__':
    main()
