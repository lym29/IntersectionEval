from pyparsing import List
import trimesh
import numpy as np
import os


# teeth_adj_list = {"17": ["16"],  
#                   "16": ["15", "17"],
#                   "15": ["14", "16"],}

teeth_order = {"upper": [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27],
               "lower": [37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47]}


def load_teeth_files(file_path):
    mesh_dict = {}
    for i in [1, 2]:
        for j in range(7, 0, -1):
            if (i == 1):
                index = j
            if (i == 2):
                index = 8 - j
            prex_upper = i
            prex_lower = i+2

            upper_fname = f"{file_path}/{prex_upper}{index}.ply"
            lower_fname = f"{file_path}/{prex_lower}{index}.ply"

            if os.path.exists(upper_fname):
                mesh = trimesh.load_mesh(upper_fname)
                prex = prex_upper
                is_exist = True
            elif os.path.exists(lower_fname):
                mesh = trimesh.load_mesh(lower_fname)
                prex = prex_lower
                is_exist = True
            else:
                is_exist = False
            
            if is_exist is False:
                pass
            mesh_dict[f"{prex}{index}"] = mesh

    return mesh_dict