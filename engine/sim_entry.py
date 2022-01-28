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

import sys
import json



def load_ply_mesh(fn, scale, offset):
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

MAT_DICT={"elastic":MPMSolver.material_elastic,"water":MPMSolver.material_water}

@ti.data_oriented
class SimEntry():
    def __init__(self,conf_fn):
      ti.init(arch=ti.cuda, kernel_profiler=True, device_memory_fraction=0.85)
      # switch to cpu if needed

      json_data=None
      with open(conf_fn, 'r', encoding='utf8')as fp:
          json_data = json.load(fp)
      if json_data is None:
          raise Exception('config  json is none! ')
      self.mesh_ids = []
      self.output = json_data["output"]
      solver_param=json_data["mpm_solver"]
      _res=tuple(solver_param["res"])
      _size=solver_param["size"]
      _unbounded=bool(solver_param["unbounded"])
      _water_density= solver_param["water_density"]
      _dt_scale= solver_param["dt_scale"]
      self.mpm=MPMSolver(res=_res, size=_size, unbounded=_unbounded, water_density=_water_density,dt_scale=_dt_scale)
      self.iter_num =1
      if "surface_collider" in json_data:
          self._make_surface_collider(json_data["surface_collider"])

      if "gravity" in json_data:
          self.mpm.set_gravity(tuple(json_data["gravity"]))
      if "meshs" in json_data:
          self._add_meshs(json_data["meshs"])
      if "cubes" in json_data:
          self._add_cubes(json_data["cubes"])
      if "iter_num" in json_data:
          self.iter_num= json_data["iter_num"]


    def _make_surface_collider(self,params):
        surface_type={"separate":self.mpm.surface_separate}
        _point = tuple(params["point"])
        _normal = tuple(params["normal"])
        _surface = surface_type[ params["surface"]]
        _friction =params["friction"]
        self.mpm.add_surface_collider(_point,_normal,_surface,_friction)
    def _add_meshs(self,params):
        color = 155 * 65536 + 55 * 256 + 205
        for p in params:
            filename=p["filename"]
            _scale=p["scale"]
            _offset=p["offset"]
            _velocity=p["velocity"]
            _translation=p["translation"]
            _sample_density=p["sample_density"]
            _material=MAT_DICT[p["material"]]
            meshs=None
            _fn_type="ply"
            if filename.endswith(".ply"):
                meshs = load_ply_mesh(filename, scale=_scale, offset=_offset)
            elif filename.endswith(".obj"):
                _fn_type="obj"
                meshs = load_obj_mesh(filename, scale=_scale, offset=_offset)

            if meshs is not None:
                mesh_id = self.mpm.add_mesh(meshs=meshs,
                                       material=_material,
                                       color=color,
                                       sample_density=_sample_density,
                                       velocity=_velocity,
                                       fn_type=_fn_type,
                                       translation=_translation)
                self.mesh_ids.append(mesh_id)
    def _add_cubes(self,params):
        for p in params:
            _lower_corner=p["lower_corner"]
            _cube_size=p["cube_size"]
            _sample_density=p["sample_density"]
            _velocity=p["velocity"]
            _material=MAT_DICT[p["material"]]
            self.mpm.add_cube(lower_corner=_lower_corner,
                     cube_size=_cube_size,
                     color=0x14FF22,
                     material=_material,
                     sample_density=_sample_density,
                     velocity=_velocity)

    def run(self,out_base_dir,callback):
        output_dir = create_output_folder(out_base_dir + '/sim')
        for frame in range(self.iter_num):
            print(f'frame {frame}')
            t = time.time()
            self.mpm.step(2e-3, print_stat=False)
            mesh_id=0
            out_files=[]
            for p in self.output:

                _type=p["type"]
                _scale = 1
                _offset = None
                if "scale" in p:
                    _scale = p["scale"]
                if "offset" in p:
                    _offset = p["offset"]
                _zoom=None
                if "zoom" in p:
                    _zoom = p["zoom"]
                tem_name=p["tem_name"]
                out_filename=tem_name.format(frame=frame)
                full_fn = output_dir + "/" + out_filename
                if _type=="mesh":
                    self.mpm.export_mesh_obj(self.mesh_ids[mesh_id],full_fn,_scale,_offset,_zoom)
                    mesh_id+=1
                    out_files.append(out_filename)
                elif _type=="mc_mesh":
                    self.mpm.export_mc_mesh(full_fn, _scale)
                    out_files.append(out_filename)

            if callback is not None:
                callback(output_dir, frame, out_files)