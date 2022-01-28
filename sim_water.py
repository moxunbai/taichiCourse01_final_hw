import taichi as ti
import math
import time
import numpy as np
from plyfile import PlyData, PlyElement
import os
import utils
from utils import create_output_folder
from engine.mpm_solver import MPMSolver
from loader.objloader import *


from engine.mesh_io import write_obj

# with_gui = True
with_gui = False
# write_to_disk = True
write_to_disk = False

faces=None
num_vets=0
num_tris=0


# Try to run on GPU
ti.init(arch=ti.cuda, kernel_profiler=True, device_memory_fraction=0.85)
# ti.init(arch=ti.cuda, kernel_profiler=True, device_memory_GB=3)

max_num_particles = 10000000
mesh_faces = ti.Vector.field(3, dtype=ti.i32 )
obj_normals = ti.Vector.field(3, dtype=ti.f32 )
if with_gui:
    gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)

def load_mesh(fn, scale, offset):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)

    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    global num_vets
    num_vets = len(x)
    # triangles = np.zeros((num_tris, 9), dtype=np.float32)
    vertexs = np.zeros((num_vets, 3), dtype=np.float32)
    global faces
    faces = np.zeros((num_tris, 3), dtype=np.int)

    for i in range(num_vets):
        vertexs[i]=[x[i]* scale + offset[0],y[i]* scale + offset[1],z[i]* scale + offset[2]]
    for i, face in enumerate(elements['vertex_indices']):
        faces[i]=face
    # for i, face in enumerate(elements['vertex_indices']):
    #     assert len(face) == 3
    #     for d in range(3):
    #         triangles[i, d * 3 + 0] = x[face[d]] * scale + offset[0]
    #         triangles[i, d * 3 + 1] = y[face[d]] * scale + offset[1]
    #         triangles[i, d * 3 + 2] = z[face[d]] * scale + offset[2]

    print('loaded')

    # return triangles
    return {"vertexs":vertexs,"faces":faces}

def load_obj_mesh(fn, scale, offset):
    objLoader = OBJ(fn)
    assert (len(objLoader.faces) > 0)

    vexs = objLoader.vertices
    global num_tris
    num_tris = len(objLoader.faces)

    global num_vets
    num_vets = len(vexs)
    # triangles = np.zeros((num_tris, 9), dtype=np.float32)
    vertexs = np.zeros((num_vets, 3), dtype=np.float32)
    global faces
    faces = np.zeros((num_tris, 3), dtype=np.int)

    for i in range(num_vets):
        vertexs[i]=[vexs[i][0]* scale + offset[0],vexs[i][1]* scale + offset[1],vexs[i][2]* scale + offset[2]]
    for i, ele in enumerate(objLoader.faces):

        face, norms, texcoords, material=ele

        faces[i]=[face[0] ,face[1] ,face[2] ]

    print('loaded')

    # return triangles
    return {"vertexs":vertexs,"faces":faces,"texcoords":objLoader.texcoords}

meshs=None
# meshs = load_mesh('./data/models/paper_plan/paper_plane.ply', scale=0.004, offset=(0.3, 0.6, 0.5))
meshs = load_obj_mesh('./data/models/spot/spot_triangulated_good.obj', scale=0.15, offset=(0.3, 0.6, 0.2))
if meshs is not None:
   ti.root.dense(ti.i, len(meshs['faces'])).place(mesh_faces)
   ti.root.dense(ti.i, num_vets).place(obj_normals)
else:
   ti.root.dense(ti.i, 1).place(obj_normals)
   ti.root.dense(ti.i, 1).place(mesh_faces)
# Use 512 for final simulation/render
R = 128

mpm = MPMSolver(res=(R, R, R), size=1, unbounded=False, water_density=1.0,dt_scale=1)

mpm.add_surface_collider(point=(0, 0, 0),
                         normal=(0, 1, 0),
                         surface=mpm.surface_separate,
                         friction=0.5)

mpm.set_gravity((0, -25, 0))

@ti.kernel
def gen_vertex_normal(num_faces: ti.i32,num_vets: ti.i32,
                           mesh_vets: ti.ext_arr() ):

    for i in range(num_vets):
        vertexNormal = ti.Vector([0.0, 0.0, 0.0])

        totalArea = 0.0

        for j in range(num_faces):

            f = mesh_faces[j]
            p1 = ti.Vector([mesh_vets[f[0],0],mesh_vets[f[0],1],mesh_vets[f[0],2] ])
            p2 = ti.Vector([mesh_vets[f[1],0],mesh_vets[f[1],1],mesh_vets[f[1],2] ])
            p3 = ti.Vector([mesh_vets[f[2],0],mesh_vets[f[2],1],mesh_vets[f[2],2] ])
            e1 = p2 - p1
            e2 = p3 - p1
            # angle = ti.acos(e1.dot(e2))

            area = e1.cross(e2).norm() * 0.5
            vertexNormal += e1.cross(e2).normalized() * area*10000
            # if i == 0:
            #     print("vertexNormal", vertexNormal)
            totalArea += area*10000

        if totalArea > 0.0:
            obj_normals[i] = (vertexNormal / totalArea).normalized()


def visualize(particles):
    np_x = particles['position'] / 1.0
    mesh_vetx_idx = particles['mesh_vetx_idx']
    mesh_vetx_start = mesh_vetx_idx[0]
    np_x =  np_x[mesh_vetx_start:(mesh_vetx_start + num_vets), :]

    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])

    screen_pos = np.stack([screen_x, screen_y], axis=-1)
    p_colors=particles['color']
    p_colors = p_colors[mesh_vetx_start:(mesh_vetx_start + num_vets)]
    print(p_colors[0])

    gui.circles(screen_pos, radius=0.8, color= p_colors )
    gui.show()


counter = 0

start_t = time.time()

color = 155 * 65536 + 55 * 256 + 205

mesh_id=mpm.add_mesh(meshs=meshs,
                     material=MPMSolver.material_elastic,
                     color=color,
                     sample_density=4,
                     velocity=(0, -10, 0),
                     fn_type="obj",
                     translation=(( 0.2) * 0.25, 0.5, (1) * 0.1) )
mpm.add_cube(lower_corner=[0.0, 0.0, 0.0],
                     cube_size=[1.0, 0.5, 1.0],
                     color=0x14FF22,
                     material=MPMSolver.material_water,
                     sample_density=4,
                     velocity=[0.0, 0.0,0.0])
trans=np.asarray([   0.06, 0.31, 0.12])
trans2=np.asarray([ 0.3,  0.6, 0.5])

if write_to_disk:
    output_dir = create_output_folder( 'out/sim')
# faces_obj=[(faces[i,0]+1,faces[i,1]+1,faces[i,2]+1) for i in range(len(faces))]
faces_obj=[(faces[i,0] ,faces[i,1] ,faces[i,2] ) for i in range(len(faces))]
for frame in range(15000):
# for frame in range(1):
    print(f'frame {frame}')
    t = time.time()

    mpm.step(2e-3, print_stat=False)
    if with_gui and frame % 3 == 0:
        particles = mpm.particle_info()
        visualize(particles)

    if write_to_disk :

        filename = f'{frame:05d}.obj'

        spot_fn=output_dir + "/" + filename
        water_fn = output_dir + "/"+f'mc_{frame:05d}.obj'
        mpm.export_mesh_obj(mesh_id,spot_fn,100)
        mpm.export_mc_mesh(water_fn,100)
    print(f'Frame total time {time.time() - t:.3f}')
    print(f'Total running time {time.time() - start_t:.3f}')
