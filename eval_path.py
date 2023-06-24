
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
        rot, trans = rigid_transform_3D(src_v_list[i].T, tgt_v_list[i].T)
        rot = R.from_matrix(rot)
        rot = rot.as_quat()
        straight_length_list.append(np.linalg.norm(trans))

        dir_deviation = np.array([0.0,0.0,0.0]) # x,y,z euler angle
        last_center = np.mean(src_v_list[i], axis=0)
        path_len = 0
        for t in range(1,20):
            # compute path length
            verts_t = verts_dict[f"time_{t}"][i]
            center_t = np.mean(verts_t, axis=0)
            trans_t =  center_t - last_center
            last_center = center_t
            path_len += np.linalg.norm(trans_t)

            # dir 
            rot_t, _ = rigid_transform_3D(src_v_list[i].T, verts_t.T)
            rot_t = R.from_matrix(rot_t)
            rot_t = rot_t.as_quat()
            dir_ideal = geometric_slerp(R.identity().as_quat(), rot, t/21.0, tol=1e-07)
            dir_diff = R.inv(R.from_quat(dir_ideal)).as_quat() * rot_t
            dir_deviation += np.abs(R.from_quat(dir_diff).as_euler('xyz'))

        path_len += np.linalg.norm( np.mean(tgt_v_list[i],axis=0) - last_center)

        dir_deviation_list.append(dir_deviation)
        total_path_length_list.append(path_len)

    straight_len = np.mean(straight_length_list)
    dir_dev = np.mean(dir_deviation_list, axis=0)
    total_len =  np.mean(total_path_length_list)

    # avg for each teeth
    out_dir = {"avg_straight_length": straight_len,
               "total_path_length":total_len,
                "dir deviation x": dir_dev[0],
                 "dir deviation y": dir_dev[1],
                 "dir deviation z": dir_dev[2] }
    return out_dir



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
    results_dir = "/Users/yumeng/Working/results/teeth_seq/raw_0623"
    out_dir = "/Users/yumeng/Working/results/teeth_seq/eval_0623/"
    os.makedirs(out_dir, exist_ok=True)

    eval(results_dir, out_dir)
