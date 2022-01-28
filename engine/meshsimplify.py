import taichi as ti
import math
import numpy as np
from time import time
import heapq
import sys
import gc

@ti.func
def solve_equations111( m  ):
    mat=m
    x=ti.Vector([0.0, 0.0, 0.0, -1.0])
    y = ti.Vector([0.0, 0.0, 0.0, 1.0])
    if mat.determinant() != 0.0:
        x = mat.inverse() @ y
    return x

@ti.func
def solve_equations(m ):
    y = ti.Vector([0.0, 0.0, 0.0, 1.0])
    for i in ti.static(range(4)):
        j = -1
        for k in ti.static(range(4)):
            if abs(m[i, k]) < 1e-8:
                # j+=1
                continue
            else:
                j = k
                break
        # while j < 4 and abs( m[i,j]) < 1e-8:
        #     j+=1
        if j==-1:
            continue
        for k in ti.static(range(4)):
           if k != i:
              rate = m[k,j] / m[i,j]
              for l  in ti.static(range(4)):
                 m[k,l] -= m[i,l] * rate
              y[k] -= y[i] * rate

    x=ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i in ti.static(range(4)):
        j = -1
        for k in ti.static(range(4)):
            if abs(m[i, k]) < 1e-8:
                # j+=1
                continue
            else:
                j=k
                break
        # while j < 4 and abs(m[i,j]) < 1e-8:
        #     j +=1
        if j == -1:
            x  =ti.Vector( [0.0, 0.0, 0.0, -1.0])
            break
        x[i]= y[i] / m[i,j]

    return x




class EdgeHeap:
    def __init__(self):
        self.cntEdge=0
        self.heap=[]
        self.isDeleted=set()
        # self.edge_dict={}
    def makeEdgeIdenKey(self,e):
        u = min(e.v1, e.v2)
        v = max(e.v1, e.v2)
        return "%d_%d"%(u,v)
    def makeEdgeNormlKey(self,e):

        return "%d_%d"%(e.v1, e.v2),"%d_%d"%(e.v2, e.v1)
    def addEdge(self,e):
        self.cntEdge+=1
        e.id=self.cntEdge
        heapq.heappush(self.heap, e)
        ekey=self.makeEdgeIdenKey(e)
        # self.edge_dict[self.makeEdgeNormlKey(e)[0]]=e
        if ekey  in self.isDeleted:
            self.isDeleted.remove(ekey)
           # del self.isDeleted[ekey]
    def getMinDelta(self):
        e=None
        if len(self.heap) == 0:
            return None
        while self.heap:
            _e=heapq.heappop(self.heap)
            if self.makeEdgeIdenKey(_e) not in self.isDeleted :
                e=_e
                break

        return e

    def delEdge(self,e):
        # e.deltaV=float("-inf")
        # normlKey=self.makeEdgeNormlKey(e)
        # if normlKey[0] in self.edge_dict:
        #   self.edge_dict[normlKey[0]].deltaV=float("-inf")
        #   heapq.heapreplace(self.heap, self.edge_dict[normlKey[0]])
        #   del self.edge_dict[normlKey[0]]
        # if normlKey[1] in self.edge_dict:
        #   self.edge_dict[normlKey[1]].deltaV=float("-inf")
        #   heapq.heapreplace(self.heap, self.edge_dict[normlKey[1]])
        #   del self.edge_dict[normlKey[1]]
        self.isDeleted.add(self.makeEdgeIdenKey(e))
        # del e


class Edge(object):
    def __init__(self,v1,v2):
        self.deltaV=0
        self.v1=v1
        self.v2=v2
        self.id=-1
        self.v=None

    def __lt__(self,other):#operator <
        return self.deltaV < other.deltaV


@ti.data_oriented
class MeshSimplify:
    MAX_EDGES = 500000
    MAX_VERTEXS = 200000
    MAX_VEX_CONNS = 280
    def __init__(self, ratio):

       self.ratio=ratio
       self.nDelFace = 0

       self.vertexs=None

       self.vex_field = ti.Vector.field(3,dtype=ti.f32)
       # self.vex_del_field = ti.field(dtype=ti.i32)
       self.vex_conn_field = ti.field(dtype=ti.i32)
       self.edge_idx = ti.field(dtype=ti.i32,shape=())
       self.edge_field = ti.Struct.field({"v1": ti.i32, "v2": ti.i32, "deltaV": ti.f32,"v":ti.types.vector(3, ti.f32) })
       self.edge_temp_field = ti.Struct.field({"v1": ti.i32, "v2": ti.i32, "deltaV": ti.f32,"v":ti.types.vector(3, ti.f32) },shape=())
       # self.edge_temp_field = ti.Struct.field({"v1": ti.i32, "v2": ti.i32, "deltaV": ti.f32,"v":ti.types.vector(3, ti.f32) },shape=(MeshSimplify.MAX_VEX_CONNS))
       self.edge_snode=ti.root.pointer(ti.i, MeshSimplify.MAX_EDGES)

       self.vex_snode=ti.root.pointer(ti.i, MeshSimplify.MAX_VERTEXS//1024)

       self.vex_snode.pointer(ti.i,1024).place(self.vex_field )
       # self.vex_snode.pointer(ti.i,1024).place(self.vex_field,self.vex_del_field)
       self.vex_conn_snode = ti.root.pointer(ti.i, MeshSimplify.MAX_VERTEXS)
       self.vex_conn_snode.dense(ti.j,MeshSimplify.MAX_VEX_CONNS).place(self.vex_conn_field)

       self.edge_snode.place(self.edge_field)

       self.vex_connects={}
       # self.isDeleted={}
       # self.isDeleted=set()

       self.eHeap =EdgeHeap()


    def input(self,vertexs,faces):
        cntv = len(vertexs)
        cntf = len(faces)
        self.vertexs=vertexs

        self.nDelFace = (int)(self.ratio * cntf)

        self.edge_idx[None]=0
        self.eHeap = EdgeHeap()
        self.vex_conn_snode.deactivate_all()
        self.vex_snode.deactivate_all()
        self.edge_snode.deactivate_all()
        self.set_vexs_field(cntv,np.asarray(vertexs))
        for i in range(cntf):
            face = faces[i]
            a = face[0]
            b = face[1]
            c = face[2]
            self.add_vex_connects(a, b)
            self.add_vex_connects(a, c)
            self.add_vex_connects(b, c)

        tmp_vconn = np.zeros((cntv,MeshSimplify.MAX_VEX_CONNS), dtype=np.int)
        for i in range(1,cntv+1,1):
            if i in self.vex_connects:
              j=0

              for k in self.vex_connects[i]:
                  if i < k:
                    break

                  tmp_vconn[i-1,j]=k
                  j+=1

        print("cntv",cntv)
        self.set_vex_conn_field(cntv,tmp_vconn)
        self.cal_vdeltav(cntv)

        edge_arr=self.edge_field.to_numpy()

        for i in range(len(edge_arr["v1"])):
            v1=edge_arr["v1"][i]
            v2=edge_arr["v2"][i]
            v=edge_arr["v"][i]
            deltaV=edge_arr["deltaV"][i]
            if v1 >0:
                e = Edge(v1 , v2 )
                e.deltaV=deltaV

                e.v=v
                self.eHeap.addEdge(e)

        del tmp_vconn

    @ti.kernel
    def cal_vdeltav(self,m:ti.i32):
        # for i,j in self.vex_conn_field:
        for i  in range(m):
          # if self.vex_conn_field[i,0]>0:
             for j  in range(MeshSimplify.MAX_VEX_CONNS):
               if self.vex_conn_field[i,j]>0:
                  eid=ti.atomic_add(self.edge_idx[None],1)
                  self.calVAndDeltaV(i+1,self.vex_conn_field[i,j],eid )
                  # self.calVAndDeltaV(i+1,self.vex_conn_field[i,j],eid,0)
    # @ti.kernel
    # def deactivate_vex_conn(self):
    #     for i,j in self.vex_conn_field:
    #         ti.deactivate(self.vex_conn_snode,[i,j])

    @ti.kernel
    def cal_vdeltav_single(self, vid1: ti.i32, vid2: ti.i32):
        if 1==1:
           self.calVAndDeltaV(vid1, vid2, -1)

    @ti.kernel
    def cal_vdeltav_dync(self, vid: ti.i32,m: ti.i32, vex_conn_arr: ti.ext_arr()):
        for i in range(m):
           self.calVAndDeltaV(vid, vex_conn_arr[i], i,1)


    @ti.kernel
    def set_vex_conn_field(self,m: ti.i32,arr: ti.ext_arr()):
        # self.vex_conn_field[0, 0] =12
        for i in range(m):
            for j in ti.static(range(MeshSimplify.MAX_VEX_CONNS)):
                # if arr[i,j]>0:
                    self.vex_conn_field[i,j]=arr[i,j]

    @ti.kernel
    def update_vex_conn_field(self, m: ti.i32,  arr_idx: ti.ext_arr(),  arr_update: ti.ext_arr()):
        for i in range(m):
            idx = arr_idx[i] - 1
            for j in range(MeshSimplify.MAX_VEX_CONNS) :

                self.vex_conn_field[idx,j]=arr_update[i,j]
    @ti.kernel
    def update_vex_conn_field111(self, m1: ti.i32, m2: ti.i32, arr_add: ti.ext_arr(), arr_removed: ti.ext_arr()):
        for i, j in self.vex_conn_field:
            for k in range(m2):
                idx = arr_removed[k, 0] - 1
                if idx == i:
                    if self.vex_conn_field[i, j] == arr_removed[k, 1]:
                        self.vex_conn_field[i, j] = 0

        for i,j in self.vex_conn_field:
            for k in range(m1):
                idx = arr_add[k, 0] - 1
                if idx==i:
                    if self.vex_conn_field[i, j] == arr_add[k, 1]:
                        break
                    if self.vex_conn_field[i, j] == 0:
                        self.vex_conn_field[i, j] = arr_add[k, 1]
                        break


    @ti.kernel
    def set_vexs_field(self,m: ti.i32,arr: ti.ext_arr()):
        for i in range(m):
            self.vex_field[i]=ti.Vector([arr[i,0],arr[i,1],arr[i,2]])
    @ti.func
    # def calVAndDeltaV(self,v1,v2,eid,etype):
    def calVAndDeltaV(self,v1,v2,eid ):

        mat = self.calVertexDelta( v1) + self.calVertexDelta( v2)
        v  = self.calVertexPos(v1,v2, mat)
        # if eid<0:
        #     print("v1",self.vex_field[v1-1])
        #     print("v2",self.vex_field[v2-1])
        #     print("v",v)
        pri = 0.0
        X = ti.Vector([v[0], v[1], v[2], 1.0])
        pri = (mat @ X).dot(X)
        # if self.getCommonVertexNum(v1,v2)==2:
        #    X=ti.Vector([v[0], v[1], v[2], 1.0])
        #
        #    # for i in ti.static(range(4)):
        #    #    p = 0.0
        #    #    for j in ti.static(range(4)):
        #    #      p += X[j] * mat[i,j]
        #    #    pri += p * X[i]
        #    pri= (mat@X).dot(X)
           # x=mat@ti.Vector([v[0], v[1], v[2], 1.0])

        if eid>=0:
           # self.edge_field[eid].deltaV =x[0]+x[1]+x[2]+x[3]
           self.edge_field[eid].deltaV =pri
           self.edge_field[eid].v1 =v1
           self.edge_field[eid].v2 =v2
           self.edge_field[eid].v = v

        else:
           self.edge_temp_field[None].deltaV =pri
           self.edge_temp_field[None].v1 = v1
           self.edge_temp_field[None].v2 = v2
           self.edge_temp_field[None].v = v
           # print(pri)
           # if pri==0.0:
           #     print(pri)
           # self.edge_temp_field[eid].deltaV =pri
           # self.edge_temp_field[eid].v1 = v1
           # self.edge_temp_field[eid].v2 = v2
           # self.edge_temp_field[eid].v = v



    @ti.func
    def getCommonVertexNum(self,u,v):
        cnt = 0
        for i in  range(MeshSimplify.MAX_VEX_CONNS) :
        # for i in ti.static(range(MeshSimplify.MAX_VEX_CONNS)):
            it =self.vex_conn_field[u-1,i]
            if it>0 and self.vex_is_conn(it,v):
                cnt += 1

        return cnt


    @ti.func
    def vex_is_conn(self, v1id, v2id):
        rst = False
        for i in range(MeshSimplify.MAX_VEX_CONNS):
            if self.vex_conn_field[v2id - 1, i] == v1id:
                rst = True
                break
        return rst

    @ti.func
    def calVertexDelta(self, vid):
        ans = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        p = self.vex_field[vid - 1]
        for i in range(MeshSimplify.MAX_VEX_CONNS):
            it1 = self.vex_conn_field[vid - 1, i]
            if it1 > 0:
                for j in range(MeshSimplify.MAX_VEX_CONNS):
                    it2 = self.vex_conn_field[vid - 1, j]
                    if it2 > 0 and it1 < it2 and self.vex_is_conn(it2, it1):
                        v1 = self.vex_field[it1 - 1]
                        v2 = self.vex_field[it2 - 1]
                        n = (v1 - p).cross(v2 - p)
                        if n.norm() > 1e-8:
                            n = n.normalized()
                            tmp = ti.Vector([n.x, n.y, n.z, -p.dot(n)])
                            ans += tmp.outer_product(tmp)
                        # else:
                        #     n=ti.Vector([0.0,0.0,1.0])
                        # tmp = ti.Vector([n.x, n.y, n.z, -p.dot(n)])
                        # for i1 in ti.static(range(4)):
                        #     for j1 in ti.static(range(4)):
                        #         ans[i1,j1]+=tmp[i1] * tmp[j1]
                        # ans += tmp.outer_product(tmp)

        return ans


    @ti.func
    def calVertexPos(self,vid1,vid2,m):
        v1= self.vex_field[vid1-1]
        v2=self.vex_field[vid2-1]
        # if v1[0]==0.0 and v1[1]==0.0 and v2[2]==0.0:
        #     print("vid1",vid1)
        # if v2[0]==0.0 and v2[1]==0.0 and v2[2]==0.0:
        #     print("vid2",vid2)
        rst = (v1+v2) / 2.0

        m[3,0] = 0.0
        m[3,1] = 0.0
        m[3,2] = 0.0
        m[3,3] = 1.0

        ans=solve_equations(m )

        if ans[3] > 1e-8:
            rst= ti.Vector([ans[0], ans[1], ans[2]])
        else:
            maxk = 0.0
            maxcost = 0.0
            cost=0.0
            for k in range(11):
                kr=k*0.1
                mid =(1 - kr) * v1 + kr * v2
                result= ti.Vector([mid[0],mid[1],mid[2],1])
                cost= ( m@result).dot(result)
                if  cost < maxcost:
                  maxk = kr

            rst= (1 - maxk) * v1 + maxk * v2
        return rst


    def add_vex_connects(self,p1,p2):
        if p1 in self.vex_connects:
            self.vex_connects[p1].add(p2)
        else:
            self.vex_connects[p1]={p2}
        if p2 in self.vex_connects:
            self.vex_connects[p2].add(p1)
        else:
            self.vex_connects[p2]={p1}

    def del_edge(self):

        e = self.eHeap.getMinDelta()
        if e is None:
            return
            # break

        v0 = e.v
        # self.vex_field[len(self.vertexs)]=ti.Vector([v0[0],v0[1],v0[2]])
        self.vex_field[e.v1 - 1] = ti.Vector([v0[0], v0[1], v0[2]])
        # self.vertexs.append([v0[0],v0[1],v0[2]])
        self.vertexs[e.v1 - 1] = [v0[0], v0[1], v0[2]]

        # v0_id = len(self.vertexs)
        # self.vex_connects[v0_id]=set()

        connectV = set()  # pV0的邻接点
        updateConns = set()  #

        self.eHeap.delEdge(e)  # 打上边已经删除的标记
        # conn_addition = []
        # conn_remove = []
        # print("deltaV",e.deltaV)
        for it in self.vex_connects[e.v1]:
            if it != e.v2:

                self.eHeap.delEdge(Edge(it, e.v1))
                self.vex_connects[it].remove(e.v1)
                connectV.add(it)
                updateConns.add(it)

        for it in self.vex_connects[e.v2]:
            if it != e.v1:
                self.eHeap.delEdge(Edge(it, e.v2))
                self.vex_connects[it].remove(e.v2)
                connectV.add(it)
                updateConns.add(it)

        v0_id = int(e.v1)

        del self.vex_connects[v0_id]
        self.vex_connects[v0_id] = set()
        del self.vex_connects[int(e.v2)]

        updateConns.add(v0_id)
        updateConns.add(int(e.v2))
        # 将原来u，v的结点的邻接点集合中加入新点o
        for it in connectV:
            self.vex_connects[it].add(v0_id)
            self.vex_connects[v0_id].add(it)

        updateVexs=[]
        uvex_nparr=np.zeros((len(updateConns),MeshSimplify.MAX_VEX_CONNS), dtype=np.int)
        j=0

        for it in updateConns:
            updateVexs.append(it)
            if it in self.vex_connects:
                k=0

                for it2 in self.vex_connects[it]:
                    uvex_nparr[j,k]=it2
                    k+=1

            j+=1
        update_vexs_nparr=np.asarray(updateVexs)
        self.update_vex_conn_field(len(updateVexs),update_vexs_nparr ,uvex_nparr)

        del updateVexs
        del uvex_nparr
        del updateConns
        del update_vexs_nparr
        # self.isDeleted[e.v1]=True # 标记结点已经被删除
        # self.isDeleted[e.v2]=True
        # self.isDeleted.add(e.v2)
        # self.vex_del_field[e.v2 - 1] = 1
        self.vertexs[e.v2 - 1] = [0]

        for it in connectV:
            self.cal_vdeltav_single(v0_id,it)
            # e0 = Edge(v0_id,it)
            edge_rst=self.edge_temp_field.to_numpy()

            # e0 = Edge(self.edge_temp_field[None].v1, self.edge_temp_field[None].v2)
            e0 = Edge(int(edge_rst['v1']), int(edge_rst['v2']))
            e0.deltaV =float(edge_rst['deltaV'])
            # self.edge_temp_field[None].deltaV
            # pos=self.edge_temp_field[None].v
            e0.v = [edge_rst['v'][0],edge_rst['v'][1],edge_rst['v'][2]]
            self.eHeap.addEdge(e0)
            # self.edge_temp_field[None].deltaV = 0.0
            # self.edge_temp_field[None].v1 = 0
            # self.edge_temp_field[None].v2 = 0
            # self.edge_temp_field[None].v = ti.Vector([0.0, 0.0, 0.0])
    def start(self):

        for i in range(0,self.nDelFace,2): #开始删边
        # for i in range(0,4,2): #开始删边
           self.del_edge()

           if i % 10000 == 0:
             gc.collect()
        # print("self.edge_idx",self.edge_idx[None])
        #
        ti.print_kernel_profile_info()
        try:
            ti.print_memory_profile_info()
        except:
            pass


    def output(self,filename):
        cnt = 0
        cntv = 0
        cntf = 0
        old_map_new={}

        fo = open(filename, "w")

        for i in range(1,len(self.vertexs)+1,1):
            # if i in self.isDeleted:
            # if self.vex_del_field[i-1]==1:
            if len(self.vertexs[i-1])==1:
                continue
            cnt+=1
            old_map_new[i]=cnt
            v=self.vertexs[i-1]

            print("v %f %f %f" % (v[0], v[1], v[2]), file=fo)


        for i in range(1, len(self.vertexs) + 1, 1):
            if len(self.vertexs[i-1])==1  or i not in self.vex_connects:
            # if self.vex_del_field[i-1]==1  or i not in self.vex_connects:
            # if i in self.isDeleted or i not in self.vex_connects:
                continue

            for it1 in self.vex_connects[i]:
                if i >= it1:
                    continue
                for it2 in self.vex_connects[i]:
                    if it1<it2 and it2 in self.vex_connects[it1]:
                        # print("i", it1)
                        print("f %d %d %d" % (old_map_new[i], old_map_new[it1], old_map_new[it2]), file=fo)
                        cntf+=1

        fo.close()