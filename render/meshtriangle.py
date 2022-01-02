import taichi as ti
from .vector import *
from loader.objloader import *
import render.ray
from .material import Materials

import random
import numpy as np
# from bvh import BVH
from PIL import Image


@ti.func
def getVectors(matrix_val):
    # v=ti.Vector.field(n=matrix_val.n,dtype=ti.f32,shape=(3))
    # vp=[0.0,0.0]
    # if matrix_val.n==3:
    #     vp=[0.0,0.0,0.0]
    v0 = ti.Vector([0.0,0.0,0.0])
    v1 = ti.Vector([0.0,0.0,0.0])
    v2 = ti.Vector([0.0,0.0,0.0])

    for i in ti.static(range(matrix_val.m)):
        v0[i]=matrix_val[0,i]
        v1[i]=matrix_val[1,i]
        v2[i]=matrix_val[2,i]
    return v0,v1,v2

@ti.data_oriented
class Triangle:
    def __init__(self,vexs,   _norms,   _texcoords,  material):
        v0=ti.Vector(vexs[0])
        v1=ti.Vector(vexs[1])
        v2=ti.Vector(vexs[2])
        self.vertices = vexs
        self.texcoords =[[0.0,0.0],[0.0,0.0],[0.0,0.0],]
        self.material=material

        self.id = -1
        e1 = v1 - v0
        e2 = v2 - v0
        self.area = e1.cross(e2).norm() * 0.5

        # n = e1.cross(e2).normalized()
        # _n = [n[0], n[1], n[2]]
        # self.normal = ti.Matrix([_n, _n, _n])
        # # self.normal = n
        # self.normal_type = 1
        if _texcoords is not None and len(_texcoords) == 3:
            self.texcoords = _texcoords
        if _norms is None or len(_norms)==0:
          vn_or=e1.cross(e2)
          if vn_or.norm()==0.0:
              raise Exception('e1.cross(e2) =0')
          n = vn_or.normalized()
          _n=[n[0],n[1],n[2]]
          # self.normal =ti.Matrix([_n,_n,_n])
          self.normal =[_n,_n,_n]
          # self.normal = n
          self.normal_type = 1

        else:
            # self.normal =ti.Matrix(_norms)
            self.normal =_norms
            self.normal_type = 3
        # print(self.normal)
        self.box_min = [min(v0[0],v1[0],v2[0],),min(v0[1],v1[1],v2[1],),min(v0[2],v1[2],v2[2],) ]
        self.box_max = [max(v0[0],v1[0],v2[0],),max(v0[1],v1[1],v2[1],),max(v0[2],v1[2],v2[2],)]
        self.box_center = [0.5 * self.box_min[0] + 0.5 * self.box_max[0], 0.5 * self.box_min[1] + 0.5 * self.box_max[1],
                       0.5 * self.box_min[2] + 0.5 * self.box_max[2]]



    @property
    def bounding_box(self):
        # return self.box
        return self.box_min, self.box_max

    @property
    def center(self):
        # return self.box
        return self.box_center

    @ti.func
    def sample(self):
        x = ti.sqrt(ti.random())
        y = ti.random()
        v0,v1,v2=getVectors(self.vertices)
        pos = v0 * (1.0 - x) + v1 * (x * (1.0 - y)) + v2 * (x * y)
        return pos,self.normal

    @staticmethod
    @ti.func
    def computeBarycentric2D(x, y,z, v0, v1, v2):
        # e1=ti.Vector([v1.x - v2.x,v1.y - v2.y,v1.z - v2.z])
        # e2=ti.Vector([v0.x - v2.x,v0.y - v2.y,v0.z - v2.z])
        # area=e1.cross(e2).norm()
        # a1=ti.Vector([ x - v2.x, y - v2.y, z - v2.z]).cross(e1).norm()
        # a2=ti.Vector([ x - v2.x, y - v2.y, z - v2.z]).cross(e2).norm()
        # c1=a1/area
        # c2=a2/area
        # c3=1-c1-c2
        c1 = (x * (v1.y - v2.y) + (v2.x - v1.x) * y + v1.x * v2.y - v2.x * v1.y) / (
                v0.x * (v1.y - v2.y) + (v2.x - v1.x) * v0.y + v1.x * v2.y - v2.x * v1.y)
        c2 = (x * (v2.y - v0.y) + (v0.x - v2.x) * y + v2.x * v0.y - v0.x * v2.y) / (
                v1.x * (v2.y - v0.y) + (v0.x - v2.x) * v1.y + v2.x * v0.y - v0.x *  v2.y)
        c3 = (x * (v0.y - v1.y) + (v1.x - v0.x) * y + v0.x * v1.y - v1.x * v0.y) / (
                v2.x * (v0.y - v1.y) + (v1.x - v0.x) * v2.y + v0.x * v1.y - v1.x * v0.y)
        return c1, c2, c3

    @staticmethod
    @ti.func
    def getNormal(type,tria_field,p):
        n0, n1, n2 =getVectors(tria_field.normal)
        _normal=n0

        if type==3:
            v0, v1, v2 = getVectors(tria_field.vertices)
            alpha, beta, gamma = Triangle.computeBarycentric2D(p.x, p.y, p.z, v0, v1, v2)
            _normal = Triangle.interpolate(alpha, beta, gamma, n0 , n1 , n2 , 1.0).normalized()
        return _normal.normalized()

    @staticmethod
    @ti.func
    def interpolate( alpha, beta, gamma, vert1, vert2, vert3, weight):
        return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight


    @staticmethod
    @ti.func
    def makeInterpolateVal(x, y, v0, v1, v2, weight):
        alpha,beta,gamma = Triangle.computeBarycentric2D(x, y, v0, v1, v2)
        return Triangle.interpolate(alpha,beta,gamma, v0, v1, v2, weight)




@ti.data_oriented
class MeshTriangle:
    def __init__(self,filename, mat,trans=None,key=None):
        self.triangles = []
        self.trans = trans
        self.material =mat
        self.texture = None
        self.tex_width =-1
        self.tex_height =-1
        self.id=-1
        self.key=key

        objLoader = OBJ( filename)
        assert(len(objLoader.faces)>0)

        vexs = objLoader.vertices
        self.box_min = [infinity,infinity,infinity]
        self.box_max = [-infinity,-infinity,-infinity]

        for face, norms, texcoords, material in objLoader.faces:

            face_vertices=[]
            face_norms=[]
            face_texcoords=[]
            for v in face:

                vert = vexs[v-1]
                if trans is not None:
                    vert = trans.makeTrans(vert)
                face_vertices.append(vert)

                self.box_min = [min(self.box_min[0], vert[0]), min(self.box_min[1], vert[1]), min(self.box_min[2], vert[2])]
                self.box_max = [max(self.box_max[0], vert[0]), max(self.box_max[1], vert[1]), max(self.box_max[2], vert[2])]
            for v in norms:
                if v>0:
                    _n=objLoader.normals[v-1]
                    if trans is not None:
                        _n = trans.makeTrans(_n,0)
                    face_norms.append(_n)
            for v in texcoords:
                if v>0:
                    face_texcoords.append(objLoader.texcoords[v-1])
            if material is None or len(material)==0:
                material=self.material
            try:
               tria =Triangle(face_vertices,face_norms,face_texcoords,material)
               tria.id=len(self.triangles)
               self.triangles.append(tria)
            except Exception:
                continue
        self.box_center = [0.5 * self.box_min[0] + 0.5 * self.box_max[0], 0.5 * self.box_min[1] + 0.5 * self.box_max[1],
                           0.5 * self.box_min[2] + 0.5 * self.box_max[2]]
        # print("box_min",self.box_min)
        # print("box_max",self.box_max)

        self.n = len(self.triangles)

        _area=0.0
        for tri in self.triangles:
            _area +=tri.area
        self.area  = _area

    def set_texture(self,tex):
        self.texture=tex

    @property
    def bounding_box(self ):
        return self.box_min, self.box_max

    @property
    def center(self):
        return self.box_center

@ti.data_oriented
class MeshTriangles:
    def __init__(self,meshs,bvh):
        num_tris=0
        self._meshs=meshs
        self.num_mesh=len(meshs)
        _bvh_roots=[]
        self._tris_start=[]
        self._tris_n=[]
        _mesh_area=[]
        _mesh_nor_type=[]
        _tris= np.array([])
        _tri_vets=[]
        _tri_normals=[]
        _tri_texcoords=[]
        _tri_area=[]
        _mesh_has_tex=[]
        _tex_wh=[]
        _tex_idx=[]
        _num_tex_val=0
        _txt_data=None
        self.bvh=bvh
        self.key_idx={}
        for i in range(self.num_mesh):
            mesh=meshs[i]
            if mesh.key is not None:
                self.key_idx[mesh.key]=i
            else:
                self.key_idx[str(i)] = i
            _mesh_area.append(mesh.area)
            self._tris_start.append(num_tris)
            tri_n=len(mesh.triangles)
            num_tris+=tri_n
            self._tris_n.append(tri_n)
            _bvh_roots.append(bvh.add(mesh.triangles))
            normal_type=-1
            if mesh.texture is None:
                _tex_wh.append([0,0])
                _mesh_has_tex.append(0)
                _tex_idx.append(0)
            else:
                _tex_wh.append([mesh.texture.width,mesh.texture.height])
                _mesh_has_tex.append(1)
                _tex_idx.append(_num_tex_val)
                _num_tex_val+=mesh.texture.data.shape[0]
                if _txt_data is None:
                    _txt_data=mesh.texture.data
                else:
                    _txt_data=np.concatenate(_txt_data,mesh.texture.data)
            for j in range(tri_n):
                triangle=mesh.triangles[j]
                _tri_vets.append(triangle.vertices)
                _tri_normals.append(triangle.normal)
                _tri_area.append(triangle.area)
                if triangle.texcoords is not None:
                    _tri_texcoords.append(triangle.texcoords)
                if normal_type<0:
                    _mesh_nor_type.append(triangle.normal_type)
                    normal_type=triangle.normal_type

        self.triangles = ti.Struct.field(
            {"vertices": ti.types.matrix(n=3, m=3, dtype=ti.f32), "texts": ti.types.matrix(n=2, m=3, dtype=ti.f32),
             "normal": ti.types.matrix(n=3, m=3, dtype=ti.f32) , "area": ti.f32})

        self.meshs = ti.Struct.field(
            {"tris_start": ti.i32,"tris_n": ti.i32, "tex_idx": ti.i32, "tex_wh": ti.types.vector(2, ti.i32),
             "has_tex": ti.i32, "id": ti.i32,"bvh_root": ti.i32,
               "normal_type": ti.i32,"area": ti.f32})

        self.texture_data=ti.Vector.field(3, dtype=ti.i32)
        ti.root.dense(ti.i, num_tris).place(self.triangles)

        ti.root.dense(ti.i, self.num_mesh).place(self.meshs )
        if _txt_data is not None:
          ti.root.dense(ti.i, _txt_data.shape[0] ).place(self.texture_data)
          self.texture_data.from_numpy(_txt_data)
        else:
          ti.root.dense(ti.i,1).place(self.texture_data)

        self.triangles.vertices.from_numpy(np.asarray(_tri_vets))
        self.triangles.texts.from_numpy(np.asarray(_tri_texcoords))
        self.triangles.normal.from_numpy(np.asarray(_tri_normals))
        self.triangles.area.from_numpy(np.asarray(_tri_area))

        self.meshs.tris_start.from_numpy(np.asarray(self._tris_start))
        self.meshs.tris_n.from_numpy(np.asarray(self._tris_n))
        self.meshs.area.from_numpy(np.asarray(_mesh_area))
        self.meshs.normal_type.from_numpy(np.asarray(_mesh_nor_type))
        self.meshs.bvh_root.from_numpy(np.asarray(_bvh_roots))
        self.meshs.tex_wh.from_numpy(np.asarray(_tex_wh))
        self.meshs.tex_idx.from_numpy(np.asarray(_tex_idx))
        self.meshs.has_tex.from_numpy(np.asarray(_mesh_has_tex))

        if _txt_data is not None:
          del _txt_data

    def update_bykey(self, mesh ):
        if mesh.key is not None:
            idx=self.key_idx[mesh.key]
            self.update(mesh,idx)
    def update(self,mesh, idx ):
        self._meshs[idx]=mesh
        tris_start = self._tris_start[idx]
        tris_n = self._tris_n[idx]
        for i in range(self.num_mesh):
            bvh_root=self.bvh.add(self._meshs[i].triangles)
            self.meshs[i].bvh_root=bvh_root


        for i in range(tris_start,tris_start+tris_n,1):
            triangle = mesh.triangles[i-tris_start]

            self.triangles[i].vertices= triangle.vertices
            if triangle.texcoords is not None:
                self.triangles[i].texts = triangle.texcoords
            self.triangles[i].normal=triangle.normal
            self.triangles[i].area=triangle.area

    @ti.func
    def get_mesh(self, i ):
        return self.meshs[i]

    @ti.func
    def get_triangle(self, i ):
        return  self.triangles[i]

    @ti.func
    def getTextureColor(self,p,mesh_idx, tri_index):
        tri=self.triangles[tri_index]
        _vertices = tri.vertices
        _texs = tri.texts
        mesh=self.meshs[mesh_idx]
        tx0, tx1, tx2 = getVectors(_texs)
        v0, v1, v2 = getVectors(_vertices)
        alpha, beta, gamma = Triangle.computeBarycentric2D(p.x, p.y, p.z, v0, v1, v2)
        tx = clamp(Triangle.interpolate(alpha, beta, gamma, tx0, tx1, tx2, 1.0), 0, 1)
        twith = mesh.tex_wh[0]
        theight = mesh.tex_wh[1]
        c_idx = ti.cast((1 - tx.y) * twith, dtype=ti.i32) *twith  + ti.cast((tx.x) * theight, dtype=ti.i32) - 1 + \
                mesh.tex_idx
        return self.texture_data[c_idx]