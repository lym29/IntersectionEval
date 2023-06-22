import argparse
import json
import os

import numpy as np
import trimesh
from joblib import Parallel, delayed

# from intersection.intersection import get_sample_intersect_volume
from intersection.intersection import mesh_vert_int_exts

import pickle
from tqdm import tqdm
import open3d as o3d
import torch

import re

teeth_order = {"upper": [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27],
               "lower": [37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47]}

def eval_adjacent(mesh_1, mesh_2, scale = 0.01):

    eval_info = {}

    scaled_mesh_1 = trimesh.Trimesh(scale * mesh_1.vertices, mesh_1.faces)
    scaled_mesh_2 = trimesh.Trimesh(scale * mesh_2.vertices, mesh_2.faces)


    trimesh.repair.fix_normals(mesh_1)
    result_close, result_distance, _ = trimesh.proximity.closest_point(
        mesh_1, mesh_2.vertices
    )
    penetrating, exterior = mesh_vert_int_exts(
    mesh_1, mesh_2.vertices, result_distance
    )

    combined = trimesh.util.concatenate([mesh_1,mesh_2])
    cvxHull = trimesh.convex.convex_hull(combined)
    
    eval_info["volumes"] = mesh_1.intersection(mesh_2, engine="auto").volume
    eval_info["cvxVolDist"] = cvxHull.volume - mesh_1.volume - mesh_2.volume
    eval_info["max_depth"] = 0 if penetrating.sum() == 0 else result_distance[penetrating == 1].max()

    return eval_info

def get_meshes(time_root):
    meshes = []
    for i in range(14):
        path = os.path.join(time_root, f"{i}.ply")
        if not os.path.exists(path):
            continue
        mesh = trimesh.load(path, force='mesh')
        meshes.append(mesh)
    return meshes
            

def eval(results_dir, out_dir):
    for model_name in os.listdir(results_dir):
        print(model_name)
        root = os.path.join(results_dir, model_name)
        if os.path.isdir(root) is False:
            continue
        out_dict = {}
        for time in os.listdir(root):
            out_dict[time] = {}
            meshes = get_meshes(os.path.join(root, time))
            print(len(meshes))
            for i in range(len(meshes)-1):
                for j in [i-1, i+1]:
                    if j < 0 or j >= len(meshes):
                        continue
                    out_dict[time][f"{i}->{j}"] = eval_adjacent(meshes[i], meshes[j])
        out_path = os.path.join(out_dir, f"{model_name}.json")
        with open(out_path, 'w') as fp:
            json.dump(out_dict, fp, indent=4)
        print(out_path)


if __name__ == "__main__":

    # results_dir = "/Users/yumeng/Working/projects/MotionPlanning_GUI/results/"
    results_dir = "/Users/yumeng/Working/results/teeth_motion/selected_results"
    out_dir = "/Users/yumeng/Working/projects/MotionPlanning_GUI/eval_results/"

    eval(results_dir, out_dir)




