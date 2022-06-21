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
    v0 = ti.Vector([0.0,0.0,0.0],dt=ti.f32)
    v1 = ti.Vector([0.0,0.0,0.0],dt=ti.f32)
    v2 = ti.Vector([0.0,0.0,0.0],dt=ti.f32)

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

MAX_TRIANGLES=524288
@ti.data_oriented
class MeshTriangles:
    def __init__(self,meshs,bvh):
        self.num_tris=0
        self._meshs=meshs
        self.num_mesh=len(meshs)
        self._bvh_roots=[]
        self._tris_start=[]
        self._tris_n=[]
        self._mesh_area=[]
        self._mesh_nor_type=[]
        _tris= np.array([])
        self._tri_vets=[]
        self._tri_normals=[]
        self._tri_texcoords=[]
        self._tri_area=[]
        self._mesh_has_tex=[]
        self._tex_wh=[]
        self._tex_idx=[]
        _num_tex_val=0
        self._txt_data=None
        self.bvh=bvh
        self.key_idx={}
        self.idx_cursor=0
        for i in range(self.num_mesh):
            mesh=meshs[i]
            if mesh.key is not None:
                self.key_idx[mesh.key]=i
            else:
                self.key_idx[str(i)] = i
            self._mesh_area.append(mesh.area)
            self._tris_start.append(self.num_tris)
            tri_n=len(mesh.triangles)
            self.num_tris+=tri_n
            self._tris_n.append(tri_n)
            self._bvh_roots.append(bvh.add(mesh.triangles))
            normal_type=-1
            if mesh.texture is None:
                self._tex_wh.append([0,0])
                self._mesh_has_tex.append(0)
                self._tex_idx.append(0)
            else:
                self._tex_wh.append([mesh.texture.width,mesh.texture.height])
                self._mesh_has_tex.append(1)
                self._tex_idx.append(_num_tex_val)
                _num_tex_val+=mesh.texture.data.shape[0]
                if self._txt_data is None:
                    self._txt_data=mesh.texture.data
                else:
                    self._txt_data=np.concatenate(self._txt_data,mesh.texture.data)
            for j in range(tri_n):
                triangle=mesh.triangles[j]
                self._tri_vets.append(triangle.vertices)
                self._tri_normals.append(triangle.normal)
                self._tri_area.append(triangle.area)
                if triangle.texcoords is not None:
                    self._tri_texcoords.append(triangle.texcoords)
                if normal_type<0:
                    self._mesh_nor_type.append(triangle.normal_type)
                    normal_type=triangle.normal_type
        self.idx_cursor = self.num_tris
        self.triangles = ti.Struct.field(
            {"vertices": ti.types.matrix(n=3, m=3, dtype=ti.f32), "texts": ti.types.matrix( m=2,n=3, dtype=ti.f32),
             "normal": ti.types.matrix(n=3, m=3, dtype=ti.f32) , "area": ti.f32})

        self.meshs = ti.Struct.field(
            {"tris_start": ti.i64,"tris_n": ti.i64, "tex_idx": ti.i64, "tex_wh": ti.types.vector(2, ti.i64),
             "has_tex": ti.i64, "id": ti.i64,"bvh_root": ti.i64,
               "normal_type": ti.i64,"area": ti.f32})

        self.texture_data=ti.Vector.field(3, dtype=ti.i64)
        # ti.root.dense(ti.i, num_tris).place(self.triangles)
        # ti.root.dense(ti.i, MAX_TRIANGLES).place(self.triangles)



        # if _txt_data is not None:
        #   del _txt_data

    def setup_data_gpu(self):
        ti.root.pointer(ti.i, MAX_TRIANGLES // 1024).pointer(ti.i, 1024 // 128).pointer(ti.i, 128 // 16).pointer(ti.i,
                                                                                                                 8).place(
            self.triangles)

        ti.root.dense(ti.i, self.num_mesh).place(self.meshs)
        if self._txt_data is not None:
            ti.root.dense(ti.i, self._txt_data.shape[0]).place(self.texture_data)

        else:
            ti.root.dense(ti.i, 1).place(self.texture_data)

    def setup_data_cpu(self):

        if self._txt_data is not None:
            self.texture_data.from_numpy(self._txt_data)
        self.set_triangls(self.num_tris,np.asarray(self._tri_vets),np.asarray(self._tri_texcoords),np.asarray(self._tri_normals),np.asarray(self._tri_area))

        self.meshs.tris_start.from_numpy(np.asarray(self._tris_start))
        self.meshs.tris_n.from_numpy(np.asarray(self._tris_n))
        self.meshs.area.from_numpy(np.asarray(self._mesh_area))
        self.meshs.normal_type.from_numpy(np.asarray(self._mesh_nor_type))
        self.meshs.bvh_root.from_numpy(np.asarray(self._bvh_roots))
        self.meshs.tex_wh.from_numpy(np.asarray(self._tex_wh))
        self.meshs.tex_idx.from_numpy(np.asarray(self._tex_idx))
        self.meshs.has_tex.from_numpy(np.asarray(self._mesh_has_tex))
    @ti.kernel
    def set_triangls(self,m:ti.i64,tri_vets:ti.ext_arr(),texcoords:ti.ext_arr(),normals:ti.ext_arr(),areas:ti.ext_arr()):
        for i in range(m):
            tri_vex=ti.Matrix.zero(ti.f32,3,3)
            tri_nor=ti.Matrix.zero(ti.f32,3,3)
            tri_tex=ti.Matrix.zero(ti.f32,3,2)
            for j in ti.static(range(3)):
               for k in ti.static(range(3)):
                tri_vex[j,k]=tri_vets[i,j,k]
                tri_nor[j,k]=normals[i,j,k]
               for k in ti.static(range(2)):
                   tri_tex[j,k]=texcoords[i,j,k]
            self.triangles[i].vertices=tri_vex
            self.triangles[i].texts =tri_tex
            self.triangles[i].normal=tri_nor
            self.triangles[i].area= areas[i]
    def update_bykey(self, mesh ):
        if mesh.key is not None:
            idx=self.key_idx[mesh.key]
            self.update(mesh,idx)
    def update_mesh_list(self, meshs):
        for mesh in meshs:
            idx=self.key_idx[mesh.key]
            self.update(mesh,idx)
            if idx<len(self._tris_start)-1:
               tris_start = self._tris_start[idx]
               end_idx=tris_start+len(mesh.triangles)
               if  end_idx>self._tris_start[idx+1]:
                   self._tris_start[idx + 1]=end_idx
    def update(self,mesh, idx ):
        self._meshs[idx]=mesh
        tris_start = self._tris_start[idx]
        # tris_n = self._tris_n[idx]
        for i in range(self.num_mesh):
            bvh_root=self.bvh.add(self._meshs[i].triangles)
            self.meshs[i].bvh_root=bvh_root

        tris_n=len(mesh.triangles)
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
        tx = clamp(Triangle.interpolate(alpha, beta, gamma, tx0, tx1, tx2, 1.0), 0.0, 1.0)
        twith = mesh.tex_wh[0]
        theight = mesh.tex_wh[1]
        c_idx = ti.cast((1.0 - tx.y) * twith, dtype=ti.i64) *twith  + ti.cast((tx.x) * theight, dtype=ti.i64) - 1 + \
                mesh.tex_idx
        return self.texture_data[c_idx]