import csv
import os

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

SYNAPSE_COLORS = {
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (1.0, 1.0, 0.0),
    5: (1.0, 0.0, 1.0),
    6: (0.0, 1.0, 1.0),
    7: (1.0, 0.5, 0.0),
    8: (0.5, 0.0, 1.0),
    9: (0.0, 0.5, 1.0),
    10: (0.5, 1.0, 0.0),
    11: (1.0, 0.75, 0.8),
    12: (0.75, 0.75, 0.0),
    13: (0.5, 0.5, 0.5),
}


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert labels.max() != 0
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def _load_case(base_dir, case_idx, data_format='h5', npy_dir=None):
    if data_format == 'npy':
        img = np.load('{}/{}_image.npy'.format(npy_dir, case_idx)).astype(np.float32)
        lbl = np.load('{}/{}_label.npy'.format(npy_dir, case_idx)).astype(np.float32)
        img = np.clip(img, -75.0, 275.0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img, lbl

    image_path = base_dir + '/{}.h5'.format(case_idx)
    h5f = h5py.File(image_path, 'r')
    return h5f['image'][:], h5f['label'][:]


def _normalize_image(image):
    image = image.astype(np.float32)
    image = np.clip(image, -125.0, 275.0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def _class_overlay(base_img, mask, class_ids, alpha=0.45):
    """Overlay only target_classes with fixed SYNAPSE_COLORS (see inference_Synapse_CPS.py)."""
    rgb = np.stack([base_img, base_img, base_img], axis=-1)
    overlay = rgb.copy()
    for cls_id in sorted(class_ids):
        if cls_id not in SYNAPSE_COLORS:
            continue
        region = mask == cls_id
        if not region.any():
            continue
        color = np.array(SYNAPSE_COLORS[cls_id], dtype=np.float32)
        overlay[region] = (1 - alpha) * overlay[region] + alpha * color
    return np.clip(overlay, 0, 1)


def _select_slice_indices(length, save_all_slices, max_slices_per_case):
    if save_all_slices or max_slices_per_case < 0:
        return list(range(length))
    count = max(1, max_slices_per_case)
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def _export_case_visualizations(image_hwz, gt_hwz, pred_hwz, case_idx, vis_config):
    vis_dir = vis_config['vis_dir']
    model_tag = vis_config['model_tag']
    planes = [p.strip() for p in vis_config.get('planes', 'axial').split(',') if p.strip()]
    target_classes = [int(x) for x in vis_config.get('target_classes', '11').split(',') if x.strip()]
    save_png = vis_config.get('save_png', True)
    export_npz = vis_config.get('export_npz', False)
    save_all_slices = vis_config.get('save_all_slices', False)
    max_slices_per_case = int(vis_config.get('max_slices_per_case', -1))
    save_gt_once = vis_config.get('save_gt_once', True)

    out_dir = os.path.join(vis_dir, model_tag, case_idx)
    os.makedirs(out_dir, exist_ok=True)

    image = _normalize_image(image_hwz)
    gt = gt_hwz.astype(np.int16)
    pred = pred_hwz.astype(np.int16)

    if export_npz:
        np.savez(
            os.path.join(out_dir, f'{case_idx}_volume.npz'),
            image=image, gt=gt, pred=pred
        )

    if not save_png:
        return out_dir

    csv_rows = []
    plane_specs = {
        'axial': ('z', 0, lambda idx, img, gt, pred: (img[idx], gt[idx], pred[idx])),
        'coronal': ('y', 1, lambda idx, img, gt, pred: (img[:, idx, :], gt[:, idx, :], pred[:, idx, :])),
        'sagittal': ('x', 2, lambda idx, img, gt, pred: (img[:, :, idx], gt[:, :, idx], pred[:, :, idx])),
    }

    for plane in planes:
        if plane not in plane_specs:
            continue
        axis_name, axis_idx, getter = plane_specs[plane]
        num_slices = image.shape[axis_idx]
        slice_ids = _select_slice_indices(num_slices, save_all_slices, max_slices_per_case)

        for slice_id in tqdm(slice_ids, desc=f'{case_idx}-{plane}', leave=False):
            img_slice, gt_slice, pred_slice = getter(slice_id, image, gt, pred)
            gt_path = os.path.join(out_dir, f'{case_idx}_{plane}_slice{slice_id:03d}_input_gt.png')
            pred_path = os.path.join(out_dir, f'{case_idx}_{plane}_slice{slice_id:03d}_pred.png')
            triptych_path = os.path.join(out_dir, f'{case_idx}_{plane}_slice{slice_id:03d}_triptych.png')

            if (not save_gt_once) or (not os.path.exists(gt_path)):
                plt.imsave(gt_path, _class_overlay(img_slice, gt_slice, target_classes), cmap=None)

            plt.imsave(pred_path, _class_overlay(img_slice, pred_slice, target_classes), cmap=None)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')
            axes[1].imshow(_class_overlay(img_slice, gt_slice, target_classes))
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            axes[2].imshow(_class_overlay(img_slice, pred_slice, target_classes))
            axes[2].set_title(f'Prediction ({model_tag})')
            axes[2].axis('off')
            fig.suptitle(f'Case {case_idx} | {plane} slice {slice_id}', fontsize=12)
            fig.tight_layout()
            fig.savefig(triptych_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            for cls_id in target_classes:
                gt_mask = gt_slice == cls_id
                pred_mask = pred_slice == cls_id
                if gt_mask.sum() > 0 and pred_mask.sum() > 0:
                    dice = metric.binary.dc(pred_mask, gt_mask) * 100
                elif gt_mask.sum() == 0 and pred_mask.sum() == 0:
                    dice = 100.0
                else:
                    dice = 0.0
                csv_rows.append([case_idx, plane, slice_id, cls_id, round(dice, 2)])

    if csv_rows:
        csv_path = os.path.join(out_dir, f'{case_idx}_slice_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['case', 'plane', 'slice', 'class_id', 'dice'])
            writer.writerows(csv_rows)

    return out_dir


def test_single_case_dhc_AB(model_A, model_B, image, stride_xy, stride_z, patch_size, num_classes=1):
    image = image[np.newaxis]
    image = image.transpose(0, 3, 2, 1)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes,) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch = test_patch.transpose(2, 4)
                with torch.no_grad():
                    y1 = (model_A(test_patch) + model_B(test_patch)) / 2.0
                    y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()[0].transpose(0, 3, 2, 1)
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    score_map = score_map.transpose(0, 3, 2, 1)
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map


def validation_all_case(
    model_A, model_B, num_classes, base_dir, image_list,
    patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
    data_format='h5', npy_dir=None, vis_config=None
):
    loader = tqdm(image_list)
    total_metric = []
    for case_idx in loader:
        image, gt_mask = _load_case(base_dir, case_idx, data_format, npy_dir)
        if data_format == 'npy':
            prediction, score_map = test_single_case_dhc_AB(
                model_A, model_B, image, stride_xy, stride_z, patch_size, num_classes=num_classes
            )
            prediction = prediction.astype(np.int8)
            gt_mask = gt_mask.astype(np.int8)
            image_vis = image
        else:
            prediction, score_map = test_single_case_fast(
                model_A, model_B, image, stride_xy, stride_z, patch_size, num_classes=num_classes
            )
            prediction = torch.FloatTensor(prediction).unsqueeze(0).unsqueeze(0)
            prediction = F.interpolate(prediction, size=(160, 160, 80), mode='nearest').int()
            prediction = prediction.squeeze().numpy().astype(np.int8)
            gt_mask = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
            gt_mask = F.interpolate(gt_mask, size=(160, 160, 80), mode='nearest').int()
            gt_mask = gt_mask.squeeze().numpy().astype(np.int8)
            image_vis = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            image_vis = F.interpolate(image_vis, size=(160, 160, 80), mode='trilinear', align_corners=False)
            image_vis = image_vis.squeeze().numpy()

        if vis_config is not None:
            _export_case_visualizations(image_vis, gt_mask, prediction, case_idx, vis_config)

        if np.sum(prediction) == 0:
            case_metric = np.zeros((4, num_classes - 1))
        else:
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def validation_all_case_fast(
    model_A, model_B, num_classes, base_dir, image_list,
    patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
    data_format='h5', npy_dir=None
):
    loader = tqdm(image_list)
    total_metric = []
    for case_idx in loader:
        image, gt_mask = _load_case(base_dir, case_idx, data_format, npy_dir)
        prediction, score_map = test_single_case_fast(
            model_A, model_B, image, stride_xy, stride_z, patch_size, num_classes=num_classes
        )
        if np.sum(prediction) == 0:
            case_metric = np.zeros((1, num_classes - 1))
        else:
            case_metric = np.zeros((1, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric_dice(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def test_single_case_fast(model_A, model_B, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode='constant', constant_values=0
        )
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    bs = 0
    total_bs = 0
    image = torch.from_numpy(image.astype(np.float32)).cuda()
    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = test_patch.unsqueeze(0).unsqueeze(0)

                if bs == 0:
                    test_patches = test_patch
                    test_patches_info = {str(bs): (xs, ys, zs)}
                else:
                    test_patches = torch.cat((test_patches, test_patch), dim=0)
                    test_patches_info[str(bs)] = (xs, ys, zs)

                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
                bs += 1
                total_bs += 1
                if bs == 16 or total_bs == sx * sy * sz:
                    with torch.no_grad():
                        outputs = (model_A(test_patches) + model_B(test_patches)) / 2
                        outputs = F.softmax(outputs, dim=1)
                    outputs = outputs.cpu().data.numpy()
                    for i in range(bs):
                        output_score = outputs[i, :, :, :, :]
                        xs, ys, zs = test_patches_info[str(i)]
                        score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += output_score
                    bs = 0

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def cal_metric(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        sf = compute_surface_distances(gt, pred, spacing_mm=(1., 1., 1.))
        nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
    elif pred.sum() == 0 and gt.sum() > 0:
        dice, hd95, asd, nsd = 0, 128, 128, 0
    elif pred.sum() == 0 and gt.sum() == 0:
        dice, hd95, asd, nsd = 1, 0, 0, 1
    else:
        dice, hd95, asd, nsd = 0, 128, 128, 0
    return np.array([dice, hd95, nsd, asd])


def cal_metric_dice(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        return np.array([metric.binary.dc(pred, gt)])
    return np.zeros(1)
