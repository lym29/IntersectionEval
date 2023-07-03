
import os
import trimesh
import json
from eval_adjacent import get_meshes
from transform.utils import rigid_transform_3D
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial import geometric_slerp


def get_teeth_motion(verts_dict):
    src_v_list = verts_dict["time_0"]
    tgt_v_list = verts_dict["target"]
    dir_deviation_list = []
    total_path_length_list = []
    straight_length_list = []
    for i in range(len(src_v_list)): # num of teeth
        center_src = np.mean(src_v_list[i], axis=0)
        center_tgt = np.mean(tgt_v_list[i], axis=0)
        rot, _ = rigid_transform_3D(src_v_list[i].T, tgt_v_list[i].T)
        rot = R.from_matrix(rot)
        rot = rot.as_quat()
        straight_length_list.append(np.linalg.norm(center_src - center_tgt))

        dir_deviation = np.array([0.0,0.0,0.0]) # x,y,z euler angle
        last_center = center_src
        path_len = 0
        for t in range(1,20):
            # compute path length
            verts_t = verts_dict[f"time_{t}"][i]
            center_t = np.mean(verts_t, axis=0)
            path_len += np.linalg.norm(center_t - last_center)
            last_center = center_t

            # dir 
            rot_t, _ = rigid_transform_3D(src_v_list[i].T, verts_t.T)
            rot_t = R.from_matrix(rot_t)
            rot_t = rot_t.as_quat()
            dir_ideal = geometric_slerp(R.identity().as_quat(), rot, t/21.0, tol=1e-07)
            dir_diff = R.inv(R.from_quat(dir_ideal)).as_quat() * rot_t
            # dir_deviation += np.abs(R.from_quat(dir_diff).as_euler('xyz'))
            dir_deviation += np.linalg.norm(R.from_quat(dir_diff).as_rotvec())

        path_len += np.linalg.norm( center_tgt - last_center)

        dir_deviation_list.append(dir_deviation)
        total_path_length_list.append(path_len)

    straight_len = np.mean(straight_length_list)
    dir_dev = np.mean(dir_deviation_list)
    total_len =  np.mean(total_path_length_list)

    # avg for each teeth
    out_dir = {"avg_straight_length": straight_len,
               "total_path_length":total_len,
                "dir deviation": dir_dev,
                }
    return out_dir

def generate_linear_interp_mesh(results_dir, out_dir):
    for model_name in os.listdir(results_dir):
        print(model_name)
        root = os.path.join(results_dir, model_name)
        if os.path.isdir(root) is False:
            continue
        verts_dict = {}
        
        src_meshes = get_meshes(os.path.join(root, "time_0"))
        tgt_meshes = get_meshes(os.path.join(root, "target"))

        for i, src in enumerate(src_meshes):
            tgt = tgt_meshes[i]
            center_src = src.center_mass
            center_tgt = tgt.center_mass
            rot, _ = rigid_transform_3D(src.vertices.T, tgt.vertices.T)
            quad_src = R.identity().as_quat()
            quad_tgt = R.from_matrix(rot).as_quat()
            for t in range(1, 20):
                dir_ideal = geometric_slerp(quad_src, quad_tgt, t/21.0, tol=1e-07)
                rot_mat = R.from_quat(dir_ideal).as_matrix()
                trans_ideal = center_src + t/21.0 * (center_tgt - center_src)

                new_verts = rot_mat @ (src.vertices.T - np.expand_dims(center_src.T, axis=1)) + np.expand_dims(trans_ideal.T, axis=1)
                new_verts = new_verts.T
                new_mesh = trimesh.Trimesh(vertices=new_verts, faces=src.faces)

                out_path = os.path.join(out_dir, model_name, f"time_{t}")
                os.makedirs(out_path, exist_ok=True)

                new_mesh.export(os.path.join(out_path, f"{i}.ply"))





        



def eval(results_dir, out_dir):
    for model_name in os.listdir(results_dir):
        print(model_name)
        root = os.path.join(results_dir, model_name)
        if os.path.isdir(root) is False:
            continue
        verts_dict = {}
        for time in os.listdir(root):
            meshes = get_meshes(os.path.join(root, time))
            print(len(meshes))
            verts_dict[time] = [mesh.vertices for mesh in meshes]
        out_dict = get_teeth_motion(verts_dict)
        out_path = os.path.join(out_dir, f"{model_name}.json")
        with open(out_path, 'w') as fp:
            json.dump(out_dict, fp, indent=4)
        print(out_path)

if __name__ == "__main__":

    # results_dir = "/Users/yumeng/Working/projects/MotionPlanning_GUI/results/"
    # results_dir = "/Users/yumeng/Working/results/teeth_seq/raw_0625"
    # out_dir = "/Users/yumeng/Working/results/teeth_seq/eval_0625/"
    # os.makedirs(out_dir, exist_ok=True)

    # eval(results_dir, out_dir)

    results_dir = "/Users/yumeng/Working/results/teeth_seq/raw_0625/"
    out_dir = "/Users/yumeng/Working/results/teeth_seq/eval_0625/linear/"
    os.makedirs(out_dir, exist_ok=True)

    generate_linear_interp_mesh(results_dir, out_dir)
