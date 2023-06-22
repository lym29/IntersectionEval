import open3d as o3d

# 读取PLY文件
mesh = o3d.io.read_triangle_mesh("../model_1/11.ply")

# 计算SDF
sdf = o3d.geometry.TriangleMesh.compute_signed_distance(mesh)

# 将SDF保存为PLY文件
o3d.io.write_triangle_mesh("sdf.ply", sdf)