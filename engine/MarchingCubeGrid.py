import taichi as ti
import math
from time import time
import numpy as np
import taichi as ti
from engine.mesh_io import write_obj
from .meshsimplify import MeshSimplify

MAX_VERTEX = 3000000
EPSILON=0.000001

@ti.data_oriented
class MCGrid:
    def __init__(self, particleR, maxInGrid, maxNeighbour, mpm_solver):
        
        self.fps             = 20.0
        self.frame           = 0

        self.maxInGrid       = maxInGrid
        self.maxNeighbour    = maxNeighbour

        # self.particle_data   = particle_data
        self.mpm_solver   = mpm_solver

        # self.gridR           = particleR*0.9
        self.gridR           = particleR
        self.invGridR        = 1.0 / self.gridR
        
        self.searchR         = self.gridR  * 4.0
        self.grid_num        = 0
        # self.isolevel        = 0.5
        self.isolevel        = 0.9
        self.sample_d        = 4

        # self.gridCount       = ti.field(dtype=ti.i32)
        # self.grid            = ti.field(dtype=ti.i32)
        self.grid            = None


        self.surface_value   = ti.field( dtype=ti.f32)


        # self.debug_value     = ti.field(dtype=ti.f32)
        self.debug_value     = None
        self.vertlist        = ti.Vector.field(3, dtype=ti.f32)
        self.triangle        = ti.Vector.field(3, dtype=ti.f32)
        self.vertex_count    = ti.field(dtype=ti.i32, shape=(1))

        self.edgetable       = ti.field(dtype=ti.i32, shape=(256))
        self.tritable        = ti.field(dtype=ti.i32, shape=(256, 16))
        self.edgetablenp     = np.ones(shape=(256), dtype=np.int32)
        self.tritablenp      = np.ones(shape=(256,16), dtype=np.int32)

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.maxboundarynp   = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp   = np.ones(shape=(1,3), dtype=np.float32)
        self.blockSize       = ti.Vector.field(3, dtype=ti.i32, shape=(1))
        self.blocknp         = np.ones(shape=(1,3), dtype=np.int32)

        # self.ms = MeshSimplify(0.3)

    @ti.pyfunc
    def setup_grid_gpu(self, maxboundarynp, minboundarynp):
        
        for i in range(3):
            self.maxboundarynp[0, i]    = maxboundarynp[0, i]
            self.minboundarynp[0, i]    = minboundarynp[0, i]

        for i in range(3):
            self.blocknp[0, i]    = int(    (self.maxboundarynp[0, i] - self.minboundarynp[0, i]) / self.gridR + 1  )
        self.grid_num = int(self.blocknp[0, 0]*self.blocknp[0, 1]*self.blocknp[0, 2])

        # ti.root.dense(ti.i,   self.grid_num).place(self.gridCount)
        # ti.root.dense(ti.ij, (self.grid_num, self.maxInGrid)).place(self.grid)
        
        
        #for surface restruction
        ti.root.dense(ti.i, self.grid_num ).place(self.surface_value)


        ti.root.dense(ti.ij, (self.grid_num, 12)).place(self.vertlist)
        ti.root.dense(ti.i,  MAX_VERTEX).place(self.triangle)
        # ti.root.dense(ti.i,  self.grid_num,).place(self.debug_value)

    @ti.pyfunc
    def setup_grid_cpu(self):

        line_index = 0
        for line in open("./engine/MCdata.txt", "r"):

            if line_index < 32:
                values = line.split(',', 8)
                for i in range(len(values)-1):
                    self.edgetablenp[line_index*8+i]      = int(values[i],16)

            else:
                values = line.split(',', 16)
                for i in range(len(values)-1):
                    self.tritablenp[line_index-32, i] = int(values[i])
            line_index += 1

        # self.max_boundary.from_numpy(self.maxboundarynp)
        # self.min_boundary.from_numpy(self.minboundarynp)



        # offset = tuple(-4096 // 2 for _ in range(3))
        grid_block_size=128
        grid = ti.root.pointer(ti.ijk, self.mpm_solver.grid_size // grid_block_size)
        block = grid.pointer(ti.ijk,
                             grid_block_size // self.mpm_solver.leaf_block_size)
        # offset=tuple(-self.mpm_solver.grid_size // 2//self.sample_d for _ in range(3))
        # block.dense(ti.ijk,  self.mpm_solver.leaf_block_size//self.sample_d ).place(self.surface_value )
        # block.dense(ti.ijk,  self.mpm_solver.leaf_block_size//self.sample_d ).dense(ti.l,  12 ).place(self.vertlist )
        block.dense(ti.ijk,  self.mpm_solver.leaf_block_size  ).place(self.surface_value )
        block.dense(ti.ijk,  self.mpm_solver.leaf_block_size  ).dense(ti.l,  12 ).place(self.vertlist )

        # block.dense(ti.ijk,  2 ).place(self.surface_value)
        # block.dense(ti.ijk,  2 ).dense(ti.l,  12 ).place(self.vertlist )
        self.tri_block=ti.root.pointer(ti.i, MAX_VERTEX//1024)
        self.tri_block.pointer(ti.i, 1024//128).pointer(ti.i, 16).pointer(ti.i, 8).place(self.triangle)
        self.grid=grid

        self.blockSize.from_numpy(self.blocknp)
        self.edgetable.from_numpy(self.edgetablenp)
        self.tritable.from_numpy(self.tritablenp)
        self.max_boundary[0] = ti.Vector([1.0, 1.0, 1.0])
        self.min_boundary[0] = ti.Vector([0.0, 0.0, 0.0])
        print("MC grid szie:", self.grid_num, "MC grid R:", self.gridR)



    @ti.pyfunc
    def export_vertex(self):
        debug_value = self.debug_value.to_numpy()
        iso_value   = self.surface_value.to_numpy()

        filename = "out/" + str(self.frame)  + ".obj"
        fo = open(filename, "w")
        yz_dim  = self.blocknp[0, 1] * self.blocknp[0, 2]

        for i in range(self.grid_num):
            if iso_value[i] > 0.0:
                x = float(i // yz_dim)                        * self.gridR + self.minboundarynp[0, 0]
                y = float((i % yz_dim) // self.blocknp[0, 2]) * self.gridR + self.minboundarynp[0, 1]
                z = float(i  % self.blocknp[0, 2])            * self.gridR + self.minboundarynp[0, 2]
                print ("v %f %f %f %f %f %f" %  (x,y,z, iso_value[i], debug_value[i]/20.0, 1.0), file = fo)
        fo.close()


    # def cmp_vertex(self,v1,v2):
    #
    #     return v1[0]==v2[0]  and v1[1]==v2[1]  and v1[2]==v2[2]

    def keyof_vertex(self,v):
        return "%d_%d_%d"%(hash(v[0]),hash(v[1]),hash(v[2]))


    @ti.pyfunc
    def export_mesh(self,filename):
        vertex_count = self.vertex_count.to_numpy()
        tri_vertex   = self.triangle.to_numpy()
        tri_count    = vertex_count[0] // 3
        # tri_count    = len(tri_vertex) // 3
        # filename = "out/mc_" + str(self.frame)  + ".obj"

        if tri_count<=0:
            return
        # fo = open(filename, "w")

        new_vertexs = []
        faces = []
        vex_map_id={}
        old_map_new={}
        nr_face={}

        for  i in range(vertex_count[0]):
            # if i<len(tri_vertex):
               v=tri_vertex[i]
               vkey=self.keyof_vertex(v)
               if vkey in vex_map_id:
                   old_map_new[i]=vex_map_id[vkey]
               else:
                   vid=len(new_vertexs)
                   new_vertexs.append(v)
                   vex_map_id[vkey]=vid
                   old_map_new[i] =vid

        # for i in range(vertex_count[0]):
        #     print ("v %f %f %f" %  (tri_vertex[i, 0], tri_vertex[i, 1], tri_vertex[i, 2]), file = fo)

        # for i in range(len(new_vertexs)):
        #     print ("v %f %f %f" %  (new_vertexs[i][0], new_vertexs[i][1], new_vertexs[i][2]), file = fo)

        for i in range(tri_count):
            a=old_map_new[3*i]+1
            b=old_map_new[3*i+1]+1
            c=old_map_new[3*i+2]+1
            f=[a,b,c]
            faces.append(f)
            # f.sort()
            # fkey="%d_%d_%d"%(f[0],f[1],f[2])
            # if fkey not in nr_face:
            #    faces.append(f)
            #    nr_face[fkey]=True
            # print ("f %d %d %d" %  (a, b, c), file = fo)
        normals = np.zeros((len(new_vertexs), 3))
        self.mpm_solver.gen_vertex_normal(len(faces),len(new_vertexs),np.asarray(new_vertexs),np.asarray(faces),normals)

        write_obj(filename, new_vertexs, faces, normals, None)

        del old_map_new
        del nr_face

        # self.ms.input(new_vertexs,faces)
        # self.ms.start()
        # self.ms.output(filename)

        # fo.close()


    @ti.pyfunc
    def export_surface(self,filename,scale=1, offset=None ):
        if self.frame>0:
           self.grid.deactivate_all()
           self.tri_block.deactivate_all()
        self.update_grid()
        self.cal_surface_point()

        self.marching_cube( scale )

        self.export_mesh(filename)

        self.frame += 1



    @ti.kernel
    def update_grid(self):
        self.vertex_count[0] = 0
        # ti.deactivate(self.surface_value,())



    @ti.kernel
    def update_grid0(self):
        for i,j in self.grid:
            self.grid[i,j] = -1
            self.gridCount[i]=0
            self.vertex_count[0] = 0

        #insert pos
        for i in self.particle_data.pos:
            indexV         = ti.cast((self.particle_data.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
            
            if self.check_in_box(indexV) == 1 :
                index     = indexV.x * self.blockSize[0].y*self.blockSize[0].z + indexV.y * self.blockSize[0].z + indexV.z
                
                old = ti.atomic_add(self.gridCount[index] , 1)
                if old > self.maxInGrid-1:
                    print("mc exceed grid ", old)
                    self.gridCount[index] = self.maxInGrid
                else:
                    self.grid[index, old] = i



    @ti.kernel
    def cal_surface_point(self):
        for I in ti.grouped(self.mpm_solver.grid_m_water):
            if self.mpm_solver.grid_m_water[I]>0.0  and I[0]%self.sample_d==0 and I[1]%self.sample_d==0 and I[2]%self.sample_d==0:
            # if self.mpm_solver.grid_m[I]>0.0 :
              indexCur=I
              for m in range(-4,5):
                  for n in range(-4,5):
                      for q in range(-4,5):
                          indexNeiV = indexCur + ti.Vector([m, n, q])
                          # if self.check_in_box(indexNeiV)==1 and self.mpm_solver.grid_is_water(indexNeiV):
                          if self.check_in_box(indexNeiV)==1 :
                              self.surface_value[I//self.sample_d]+=self.mpm_solver.grid_m_water[indexNeiV]
                              # self.surface_value[I ]+=self.mpm_solver.grid_m[indexNeiV]






    @ti.kernel
    def marching_cube(self,scale:ti.f32 ):

        for I in ti.grouped(self.mpm_solver.grid_m_water):
            if I[0]%self.sample_d==0 and I[1]%self.sample_d==0 and I[2]%self.sample_d==0:
               indexCur = I
               indexCur = I//self.sample_d
               if self.check_in_box(self.sample_d*indexCur + ti.Vector([1, 1, 1])) == 1:
               # if self.check_in_box( indexCur + ti.Vector([1, 1, 1])) == 1:
                   index0 = I//self.sample_d
                   # index0 = I
                   index1 =indexCur + ti.Vector([1, 0, 0])
                   index2 =indexCur + ti.Vector([1, 1, 0])
                   index3 =indexCur + ti.Vector([0, 1, 0])
                   index4 =indexCur + ti.Vector([0, 0, 1])
                   index5 =indexCur + ti.Vector([1, 0, 1])
                   index6 =indexCur + ti.Vector([1, 1, 1])
                   index7 =indexCur + ti.Vector([0, 1, 1])


                   cubeindex = 0
                   if self.surface_value[index0] < self.isolevel:
                       cubeindex |= 1
                   if self.surface_value[index1] < self.isolevel:
                       cubeindex |= 2
                   if self.surface_value[index2] < self.isolevel:
                       cubeindex |= 4
                   if self.surface_value[index3] < self.isolevel:
                       cubeindex |= 8
                   if self.surface_value[index4] < self.isolevel:
                       cubeindex |= 16
                   if self.surface_value[index5] < self.isolevel:
                       cubeindex |= 32
                   if self.surface_value[index6] < self.isolevel:
                       cubeindex |= 64
                   if self.surface_value[index7] < self.isolevel:
                       cubeindex |= 128
                   # i=4096*4096*I[0]+4096*I[1]+I[2]

                   i=I
                   for j in range(12):
                       self.vertlist[i, j] = self.get_pos(index0)

                   if self.edgetable[cubeindex] != 0:
                       if self.edgetable[cubeindex] & 1:
                           self.vertlist[i, 0] = self.vertex_interp(self.get_pos(index0), self.get_pos(index1),
                                                                    self.surface_value[index0], self.surface_value[index1])
                       if self.edgetable[cubeindex] & 2:
                           self.vertlist[i, 1] = self.vertex_interp(self.get_pos(index1), self.get_pos(index2),
                                                                    self.surface_value[index1], self.surface_value[index2])
                       if self.edgetable[cubeindex] & 4:
                           self.vertlist[i, 2] = self.vertex_interp(self.get_pos(index2), self.get_pos(index3),
                                                                    self.surface_value[index2], self.surface_value[index3])
                       if self.edgetable[cubeindex] & 8:
                           self.vertlist[i, 3] = self.vertex_interp(self.get_pos(index3), self.get_pos(index0),
                                                                    self.surface_value[index3], self.surface_value[index0])
                       if self.edgetable[cubeindex] & 16:
                           self.vertlist[i, 4] = self.vertex_interp(self.get_pos(index4), self.get_pos(index5),
                                                                    self.surface_value[index4], self.surface_value[index5])
                       if self.edgetable[cubeindex] & 32:
                           self.vertlist[i, 5] = self.vertex_interp(self.get_pos(index5), self.get_pos(index6),
                                                                    self.surface_value[index5], self.surface_value[index6])
                       if self.edgetable[cubeindex] & 64:
                           self.vertlist[i, 6] = self.vertex_interp(self.get_pos(index6), self.get_pos(index7),
                                                                    self.surface_value[index6], self.surface_value[index7])
                       if self.edgetable[cubeindex] & 128:
                           self.vertlist[i, 7] = self.vertex_interp(self.get_pos(index7), self.get_pos(index4),
                                                                    self.surface_value[index7], self.surface_value[index4])
                       if self.edgetable[cubeindex] & 256:
                           self.vertlist[i, 8] = self.vertex_interp(self.get_pos(index0), self.get_pos(index4),
                                                                    self.surface_value[index0], self.surface_value[index4])
                       if self.edgetable[cubeindex] & 512:
                           self.vertlist[i, 9] = self.vertex_interp(self.get_pos(index1), self.get_pos(index5),
                                                                    self.surface_value[index1], self.surface_value[index5])
                       if self.edgetable[cubeindex] & 1024:
                           self.vertlist[i, 10] = self.vertex_interp(self.get_pos(index2), self.get_pos(index6),
                                                                     self.surface_value[index2],
                                                                     self.surface_value[index6])
                       if self.edgetable[cubeindex] & 2048:
                           self.vertlist[i, 11] = self.vertex_interp(self.get_pos(index3), self.get_pos(index7),
                                                                     self.surface_value[index3],
                                                                     self.surface_value[index7])


                   k = 0

                   while (self.tritable[cubeindex, k] != -1):
                       old = ti.atomic_add(self.vertex_count[0], 3)
                       if old < MAX_VERTEX:
                           self.triangle[old + 0] = self.vertlist[i, self.tritable[cubeindex, k]]*scale
                           self.triangle[old + 1] = self.vertlist[i, self.tritable[cubeindex, k + 1]]*scale
                           self.triangle[old + 2] = self.vertlist[i, self.tritable[cubeindex, k + 2]]*scale
                       else:
                           print("exceed max tri", old)
                       k += 3


    @ti.func
    def check_in_box(self, index):
        ret = 1
        # if (index.x < 0) or (index.x >= self.blockSize[0].x) or \
        #    (index.y < 0) or (index.y >= self.blockSize[0].y) or \
        #    (index.z < 0) or (index.z >= self.blockSize[0].z):
        #     ret = 0
        if (index.x < 0) or (index.x >= 4096) or \
           (index.y < 0) or (index.y >= 4096) or \
           (index.z < 0) or (index.z >= 4096):
            ret = 0
        return ret
    


    @ti.func
    def get_cell_indexV(self, index):
        yz_dim = self.blockSize[0].y * self.blockSize[0].z
        x      =  index // yz_dim
        y      = (index %  yz_dim) //self.blockSize[0].z
        z      = index  %  self.blockSize[0].z
        return ti.Vector([x, y, z])
    
    @ti.func
    def get_cell_index(self, index):
        return index.x * self.blockSize[0].y*self.blockSize[0].z + index.y * self.blockSize[0].z + index.z

    @ti.func
    def get_posV(self, indexV):
        return self.min_boundary[0] +  ti.cast(indexV, ti.f32) * self.gridR 

    @ti.func
    def get_pos(self, index):
        return self.get_posV( index *self.sample_d)
        # return self.get_posV( index )
        # return self.get_posV(self.get_cell_indexV(index))


    @ti.func
    def weight_func(self, xi, xj):
        ret = 0.0
        r   = xi - xj
        dis = r.norm() 
        if dis < self.searchR*2.0:
            ret = 1.0 - pow( r.norm() / (self.searchR*2.0), 3.0)
        return ret



    @ti.func
    def vertex_interp(self, p1, p2, valp1, valp2):
        if  self.check_pos(p2, p1) == 1:
            temp = p1
            p1 = p2
            p2 = temp  

            tmp1 = valp1
            valp1 = valp2
            valp2 = tmp1

        p = p1
        if abs(valp1 - valp2) > 0.00001:
            p = p1 + (p2 - p1) / (valp2 - valp1)*(self.isolevel - valp1)

        return p

    @ti.func
    def check_pos(self, p2, p1):
        ret = 1
        if p2.x < p1.x:
            ret = 1
        elif p2.x > p1.x:
            ret = 0

        if p2.y < p1.y:
            ret = 1
        elif p2.y > p1.y:
            ret = 0

        if p2.z < p1.z:
            ret = 1
        elif p2.z > p1.z:
            ret = 0

        return ret



