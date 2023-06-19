import argparse
import json
import os

import numpy as np
import trimesh
from joblib import Parallel, delayed

from intersection import get_sample_intersect_volume
from intersection import mesh_vert_int_exts

import pickle
from tqdm import tqdm
import open3d as o3d
import torch


def dict_for_connected_faces(faces):
    v2f = {}
    for f_id, face in enumerate(faces):
        for v in face:
            v2f.setdefault(v, []).append(f_id)
    return v2f

def geneSampleInfos_quick(data):
    device = o3d.core.Device('cuda' if torch.cuda.is_available() else 'CPU:0')
    for k in data:
        if k == "file_name":
            continue
        data[k] = torch.Tensor(np.stack(data[k]))
    hand_verts = data["hand_verts"]
    obj_verts = data["object_verts"]
    obj_faces = data["object_faces"]

# def scale_mesh(mesh:trimesh.Trimesh, scale):
#     verts = mesh.vertices - mesh.center_mass
#     verts = verts * scale + mesh.center_mass
#     return verts

def geneSampleInfos(fname_lists, mesh1_verts, mesh1_faces, mesh2_verts, mesh2_faces, scale=1):
    """
    Args:
        scale (float): convert to meters
    """
    # object_faces = [ obj_face[:, ::-1]for obj_face in object_faces]  # CCW to CW

    sample_infos = []

    for hand_vert, hand_face, obj_vert, obj_face in tqdm(zip(
        mesh1_verts, mesh1_faces, mesh2_verts, mesh2_faces
    )):
        sample_info = {
            "file_names": fname_lists,
            "hand_verts": hand_vert * scale,
            "hand_faces": hand_face,
            "obj_verts": obj_vert * scale,
            "obj_faces": obj_face,
        }

        obj_mesh = trimesh.load({"vertices": sample_info["obj_verts"], "faces": obj_face})
        trimesh.repair.fix_normals(obj_mesh)
        result_close, result_distance, _ = trimesh.proximity.closest_point(
            obj_mesh, sample_info["hand_verts"]
        )
        penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_vert, result_distance
        )
        sample_info["max_depth"] = 0 if penetrating.sum() == 0 else result_distance[penetrating == 1].max()

        sample_infos.append(sample_info)

    return sample_infos

def detect_intersect(
    sample_infos,  
    saved_path,
    workers=8
):
    os.makedirs(saved_path, exist_ok=True)

    max_depths = [sample_info["max_depth"] for sample_info in sample_infos]
    file_names = [sample_info["file_names"] for sample_info in sample_infos]

    volumes = Parallel(n_jobs=workers, verbose=5)(
        delayed(get_sample_intersect_volume)(sample_info)
        for sample_info in sample_infos
    )
    
    results_path = os.path.join(saved_path,"results.json")
    with open(results_path, "w") as j_f:
        json.dump(
            {
                "max_depths": max_depths,
                "mean_max_depth": np.mean(max_depths),

                "volumes": volumes,
                "mean_volume": np.mean(volumes),

                "file_names": file_names,
            },
            j_f,
        )
        print("Wrote results to {}".format(results_path))




if __name__ == "__main__":

    # data_path = "/ghome/l5/ymliu/results/oakink/baseline_0414/test_grasp_results/A01002/0/"
    # obj_file = os.path.join(data_path, "obj_mesh_0.ply")
    # hand_file = os.path.join(data_path, "rh_mesh_rec_0.ply")

    work_dir = "/Users/yumeng/Working/projects/MotionPlanning_GUI/"
    data_path = os.path.join(work_dir,"data", "model_1")
    saved_path = os.path.join(work_dir, "eval_results", "model_1")

    mesh1 = trimesh.load_mesh(os.path.join(data_path, "11.ply"))
    mesh2 = trimesh.load_mesh(os.path.join(data_path, "12.ply"))


    # vhacd_exe = "/ghome/l5/ymliu/3rdparty/VHACD_bin/testVHACD"

    sample_infos = geneSampleInfos(fname_lists=[data_path],
                                   mesh1_verts=[mesh1.vertices],
                                   mesh1_faces=[mesh1.faces],
                                   mesh2_verts=[mesh2.vertices],
                                   mesh2_faces=[mesh2.faces],
                                   scale=0.1)
    
    print("<---- Get Sample Info Done ---->")

    detect_intersect(sample_infos=sample_infos, saved_path=saved_path)


    print(sample_infos)




