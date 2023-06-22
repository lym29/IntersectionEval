import numpy as np
import trimesh
import torch

def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(inp.shape))
    ]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def thresh_ious(gt_dists, pred_dists, thresh):
    """
    Computes the contact intersection over union for a given threshold
    """
    gt_contacts = gt_dists <= thresh
    pred_contacts = pred_dists <= thresh
    inter = (gt_contacts * pred_contacts).sum(1).float()
    union = union = (gt_contacts | pred_contacts).sum(1).float()
    iou = torch.zeros_like(union)
    iou[union != 0] = inter[union != 0] / union[union != 0]
    return iou


def meshiou(gt_dists, pred_dists, threshs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    For each thresholds, computes thresh_ious and averages accross batch dim
    """
    all_ious = []
    for thresh in threshs:
        ious = thresh_ious(gt_dists, pred_dists, thresh)
        all_ious.append(ious)
    iou_auc = np.mean(
        np.trapz(torch.stack(all_ious).cpu().numpy(), axis=0, x=threshs)
    )
    batch_ious = torch.stack(all_ious).mean(1)
    return batch_ious, iou_auc


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
    else:
        loss = torch.Tensor([0]).cuda()
    return loss


def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def thres_loss(vals, thres=25):
    """
    Args:
        vals: positive values !
    """
    thres_mask = (vals < thres).float()
    loss = masked_mean_loss(vals, thres_mask)
    return loss


def compute_naive_contact_loss(points_1, points_2, contact_threshold=25):
    dists = batch_pairwise_dist(points_1, points_2)
    mins12, _ = torch.min(dists, 1)
    mins21, _ = torch.min(dists, 2)
    loss_1 = thres_loss(mins12, contact_threshold)
    loss_2 = thres_loss(mins21, contact_threshold)
    loss = torch.mean((loss_1 + loss_2) / 2)
    return loss


def mesh_vert_int_exts(obj1_mesh, obj2_verts, result_distance, tol=0.1):
    nonzero = result_distance > tol
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    penetrating = [sign == 1][0] & nonzero
    exterior = [sign == -1][0] & nonzero
    return penetrating, exterior


def get_contact_info(
    hand_vert,
    hand_faces,
    obj_vert,
    obj_faces,
    result_close=None,
    result_distance=None,
    contact_thresh=25,
):
    obj_mesh_dict = {"vertices": obj_vert, "faces": obj_faces}
    obj_mesh = trimesh.load(obj_mesh_dict)
    trimesh.repair.fix_normals(obj_mesh)
    # hand_mesh_dict = {'vertices': hand_vert, 'faces': hand_faces}
    # hand_mesh = trimesh.load(hand_mesh_dict)
    # trimesh.repair.fix_normals(hand_mesh)
    if result_close is None or result_distance is None:
        result_close, result_distance, _ = trimesh.proximity.closest_point(
            obj_mesh, hand_vert
        )
    penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_vert, result_distance
    )

    below_dist = result_distance < contact_thresh
    missed_mask = below_dist & exterior
    return result_close, missed_mask, penetrating


def get_depth_info(obj_mesh, hand_verts):
    result_close, result_distance, _ = trimesh.proximity.closest_point(
        obj_mesh, hand_verts
    )
    penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_verts, result_distance
    )
    return result_close, result_distance, penetrating

def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def intersect(obj_mesh:trimesh.Trimesh, hand_mesh:trimesh.Trimesh, engine="auto"):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh


def get_sample_intersect_volume(sample_info, mode="voxels"):
    hand_mesh = trimesh.Trimesh(
        vertices=sample_info["hand_verts"], faces=sample_info["hand_faces"]
    )
    obj_mesh = trimesh.Trimesh(
        vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"]
    )
    if mode == "engines":
        try:
            intersection = intersect(obj_mesh, hand_mesh, engine="scad")
            if intersection.is_watertight:
                volume = intersection.volume
            else:
                intersection = intersect(obj_mesh, hand_mesh, engine="blender")
                # traceback.print_exc()
                if intersection.vertices.shape[0] == 0:
                    volume = 0
                elif intersection.is_watertight:
                    volume = intersection.volume
                else:
                    volume = None
        except Exception:
            intersection = intersect(obj_mesh, hand_mesh, engine="blender")
            # traceback.print_exc()
            if intersection.vertices.shape[0] == 0:
                volume = 0
            elif intersection.is_watertight:
                volume = intersection.volume
            else:
                volume = None
    elif mode == "voxels":
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume

