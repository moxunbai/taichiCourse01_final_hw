import taichi as ti
from render.vector import *
import render.ray as ray
from render.material import *
from render.hittable import *
from render.meshtriangle import *
from render.camera import Camera
from render.Texture2D  import *
from render.EnvLight  import *
import random
import numpy as np
from time import time
from render.bvh import  BVHS

EPSILON=0.0001
@ti.func
def getVectors(matrix_val):
    v0 = ti.Vector([0.0,0.0,0.0],dt=ti.f32)
    v1 = ti.Vector([0.0,0.0,0.0],dt=ti.f32)
    v2 = ti.Vector([0.0,0.0,0.0],dt=ti.f32)

    for i in ti.static(range(matrix_val.m)):
        v0[i]=matrix_val[0,i]
        v1[i]=matrix_val[1,i]
        v2[i]=matrix_val[2,i]
    return v0,v1,v2

@ti.data_oriented
class Translate:
    def __init__(self,displacement):
        self.transMatrix=displacement

    def makeTrans(self,v,f=1):
        rst = ti.Vector([v[0], v[1], v[2], f])
        rst= self.transMatrix@rst
        return  [rst[0],rst[1],rst[2]]
@ti.data_oriented
class Scene:
    def __init__(self):
        self.objects = []
        self.ids = []
        self.meshs = []
        self.spheres = []
        self.objlights = []
        self.geo_types = []
        self.local_idx = []#物体在自己所属分类集合中索引
        self._light_num = 0
        self.env_light = 0
        self.bvh_root =-1
        self.bvh = -1
        self.key_idx = {}


    def add(self, object,lighttype=0):
        object.id = len(self.objects)
        self.ids.append(object.id)
        self.key_idx[object.key]=object.id
        self.objects.append(object)
        if isinstance(object, Sphere):
            self.local_idx.append(len(self.spheres))
            self.spheres.append(object)
            self.geo_types.append(2)

        elif isinstance(object, MeshTriangle):
            self.geo_types.append(1)
            self.local_idx.append(len(self.meshs))
            self.meshs.append(object)

        self.objlights.append(lighttype)
        self._light_num+=int(lighttype==1)

        self.meshs_field =None
        self.spheres_field =None

    def set_env_light(self,env_light):
        self.env_light=env_light
    def commit(self):
        ''' Commit should be called after all objects added.
            Will compile bvh and materials. '''
        self.n = len(self.objects)

        self.bvh = BVHS()
        self.materials = Materials(self.n, self.objects)

        self.bvh_root=self.bvh.add(self.objects)
        if len(self.meshs)>0:
            self.meshs_field=MeshTriangles(self.meshs,self.bvh)
        if len(self.spheres)>0:
            self.spheres_field=Spheres(self.spheres)
        else:
            self.spheres_field = Spheres([])
        self.objs_field = ti.Struct.field({"geo_type": ti.i32, "li_type": ti.i32, "local_idx": ti.i32, "id": ti.i32})


        self.light_num = ti.field(ti.i32,shape=())

        ti.root.dense(ti.i, self.n).place(self.objs_field)
        if len(self.meshs)>0:
            self.meshs_field.setup_data_gpu()
        self.materials.setup_data_gpu()


        self.light_num[None] = self._light_num

        self.objs_field.geo_type.from_numpy(np.asarray(self.geo_types))
        self.objs_field.li_type.from_numpy(np.asarray(self.objlights))
        self.objs_field.local_idx.from_numpy(np.asarray(self.local_idx))
        self.objs_field.id.from_numpy(np.asarray(self.ids))

        if len(self.meshs)>0:
            self.meshs_field.setup_data_cpu()
        if len(self.spheres)>0:
            self.spheres_field.setup_data_cpu()
        self.materials.setup_data_cpu()

        self.bvh.build()

        # del self.objects
        del self.ids
        # del self.meshs
        del self.spheres
        del self.objlights
        del self.geo_types

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    def update_obj(self,obj ):
        if isinstance(obj, MeshTriangle):
            self.bvh.clear()
            i=self.key_idx[obj.key]
            obj.id=i
            self.objects[i]=obj
            self.bvh_root = self.bvh.add(self.objects)
            self.meshs_field.update_bykey(obj )
            self.bvh.build()
    def update_mesh_list(self,mesh_list ):
        self.bvh.clear()
        for i in range(len(mesh_list)):
           obj = mesh_list[i]
           objid = self.key_idx[obj.key]
           obj.id = objid
           self.objects[objid] = obj
        self.bvh_root = self.bvh.add(self.objects)
        self.meshs_field.update_mesh_list(mesh_list)
        self.bvh.build()


    @staticmethod
    def GenMeshObj(json_data):
        is_light = 0
        trans = None
        type=json_data["type"]
        mate = Scene.GenMate(json_data)
        if "is_light" in json_data:
            is_light = json_data["is_light"]
        if "transformation" in json_data:
            trans_data = json_data["transformation"]
            trans = Translate(
                makeTransformations(trans_data["scale"], math.pi * trans_data["routeY"], trans_data["translation"]))
        obj=None
        if type == "MeshTriangle":
            mesh_fn = json_data["filename"]
            _key = None
            if "key" in json_data:
                _key = json_data["key"]
            obj = MeshTriangle(mesh_fn, mate, trans, _key)
            if "texture" in json_data:
                tex = Texture2D(json_data["texture"])
                obj.set_texture(tex)
        return obj
    @staticmethod
    def GenMate(json_data):
        mate_data = json_data["material"]
        mate_type = mate_data["type"]
        mate = None
        color = [1.0, 1.0, 1.0]
        if "color" in mate_data:
            color = mate_data["color"]

        if mate_type == 'Dielectric':
            ior = 1.5
            if "ior" in mate_data:
                ior = mate_data["ior"]
            mate = Dielectric(ior,color)
        elif mate_type == 'Metal':
            roughness = 0.0

            if "roughness" in mate_data:
                roughness = mate_data["roughness"]
            mate = Metal(color, roughness)
        elif mate_type == 'Lambert_light':

            mate = Lambert_light(color)
        else:

            mate = Lambert(color)
        return mate
    @staticmethod
    def GenScene(json_data):
        scene=Scene()
        env_light=None
        if "env_light" in json_data:
            env_light_data = json_data["env_light"]
            t1=time()
            texture_fn=None
            if "texture" in env_light_data:
                texture_fn=env_light_data["texture"]
            env_light = EnvLight(env_light_data["intensity"], env_light_data["color"] , texture_fn)
            print("Create EnvLight cost:",time()-t1)
        else:
            env_light = EnvLight()
        scene.set_env_light(env_light)
        if "models" not in json_data:
            return scene
        models = json_data["models"]
        for m in models:
            type = m["type"]
            mate =Scene.GenMate(m)
            is_light = 0
            trans = None
            if "is_light" in m:
                is_light = m["is_light"]
            if "transformation" in m:
                trans_data = m["transformation"]
                trans = Translate(
                    makeTransformations(trans_data["scale"], math.pi * trans_data["routeY"], trans_data["translation"]))

            obj = None
            if type == "MeshTriangle":
                mesh_fn = m["filename"]
                _key=None
                if "key" in m:
                    _key=m["key"]
                obj = MeshTriangle(mesh_fn, mate, trans,_key)
                if "texture" in m:
                    tex = Texture2D(m["texture"])
                    obj.set_texture(tex)
            elif type == "Sphere":
                radius = 10.0
                center = [1.0, 1.0, 1.0]
                if "radius" in m:
                    radius = m["radius"]
                if "center" in m:
                    center = m["center"]
                obj = Sphere(center, radius, mate)
            if obj is not None:
                scene.add(obj, is_light)
        return scene
    @ti.func
    def getTextureColor(self,hitRecord):
        obj_index = hitRecord.hit_index
        tri_index=hitRecord.tri_idx
        color = self.meshs_field.getTextureColor(hitRecord.p,obj_index,tri_index)
        # tri = self.triangles[tri_index]
        # tx0, tx1, tx2 = getVectors(tri.tx)
        # v0, v1, v2 = getVectors(tri.vs)
        # alpha, beta, gamma = Triangle.computeBarycentric2D(p.x, p.y, p.z, v0, v1, v2)
        # tx = clamp(Triangle.interpolate(alpha, beta, gamma, tx0, tx1, tx2, 1.0),0,1)
        # tweight=obj.texture_pro[0]
        # theight=obj.texture_pro[1]
        # c_idx=ti.cast(( 1-tx.y)*theight,dtype=ti.i32)*tweight+ti.cast(( tx.x)*tweight,dtype=ti.i32)-1+obj.texture_pro[2]

        # color = self.texture_datas[c_idx]
        return ti.cast(color,dtype=ti.f32)/255

    @ti.func
    def hit_obj(self,obj_id,ray_origin, ray_direction, t_min,closest_so_far):
        hit_anything = False
        p = Point([0.0, 0.0, 0.0])
        n = Vector([0.0, 0.0, 0.0])
        front_facing = True
        hit_index=0
        hit_tri_index=-1
        t=0.0

        obj_box=self.objs_field[obj_id]

        if obj_box.geo_type==1:
            hit_anything,t, p, n, front_facing,hit_tri_index =self.hit_meshs(obj_box,ray_origin, ray_direction, t_min,closest_so_far)

        if obj_box.geo_type==2  :
            hit_anything,t, p, n, front_facing= self.hit_sphere(obj_box,ray_origin, ray_direction, t_min,closest_so_far)

        return hit_anything,t, p, n, front_facing,hit_tri_index
    @ti.func
    def hit_meshs(self,obj_box,ray_origin, ray_direction, t_min, t_max ):
        hit_anything = False

        closest_so_far = t_max

        p = Point([0.0, 0.0, 0.0])
        n = Vector([0.0, 0.0, 0.0])
        front_facing = True
        t=0.0
        i = 0
        hit_tri_index = -1
        mesh = self.meshs_field.get_mesh(obj_box.local_idx)
        curr = mesh.bvh_root
        # obj_box.pos[2]
        mat_c,mat_type,mat_roughness,mat_ior= self.get_mattype(obj_box.id)

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id,area_id = self.bvh.get_full_id(curr)

            if obj_id != -1:
                # this is a leaf node, check the sphere

                hit, _t, _p, _n, _front_facing,_hit_tri_index = self.hit_triangle(obj_id,mesh,
                                    ray_origin, ray_direction, t_min,
                                    closest_so_far )
                # print("hit_triangle ", hit)
                valid_face=True
                if mat_type!=2 and not _front_facing:
                    valid_face=False
                if hit and valid_face:
                   hit_anything = True
                   closest_so_far = _t
                   t=_t
                   p=_p
                   n=_n
                   front_facing=_front_facing
                   hit_tri_index=_hit_tri_index

                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):

                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id

        return hit_anything, t, p, n, front_facing,hit_tri_index

    @ti.func
    def hit_triangle(self,tria_id,mesh,ray_origin, ray_direction, t_min,t_max ):
        tria_start=mesh.tris_start
        hit_tri_index=tria_start+tria_id
        # obj=self.triangles[hit_tri_index]
        triangle=self.meshs_field.get_triangle(hit_tri_index)

        # _vx=obj.vs
        _vx=triangle.vertices

        hit=True

        root = -1.0
        p = Point([0.0, 0.0, 0.0])
        n = Point([0.0, 0.0, 0.0])
        t = None
        front_facing = True
        v0, v1, v2 = getVectors(_vx)
        e1=v1 - v0
        e2=v2 - v0
        s=ray_origin-v0
        s1=ray_direction.cross( e2)
        s2=s.cross( e1)
        det=s1.dot(e1)
        if abs(det) < EPSILON:
            hit = False
        else:
            det_inv=1.0/det
            t=s2.dot(e2)*det_inv
            if t<t_min or t_max<t:
                hit = False
            else:
                b1=s1.dot(s)*det_inv
                b2=s2.dot(ray_direction)*det_inv
                b3=1-b1-b2
                if b1<0 or b1>1 or b2<0 or b2>1 or b3<0 or b3>1:
                    hit = False
                else:
                    p = ray.at(ray_origin, ray_direction, t)
                    _norm = Triangle.getNormal(mesh.normal_type,triangle, p)
                    n = _norm
                    # hit = ray_direction.dot(_norm) <= 0
                    front_facing = is_front_facing(ray_direction, n)
                    n = n if front_facing else -n

        return hit, t, p, n, front_facing,hit_tri_index

    @ti.func
    def hit_sphere(self,obj_box,ray_origin, ray_direction , t_min, t_max):
        sph=self.spheres_field.get(obj_box.local_idx)
        center=sph.center
        radius=sph.radius
        oc = ray_origin - center
        a = ray_direction.norm_sqr()
        half_b = oc.dot(ray_direction)
        c = (oc.norm_sqr() - radius ** 2)
        discriminant = (half_b ** 2) - a * c

        hit = discriminant >= 0.0
        root = -1.0
        p = Point([0.0, 0.0, 0.0])
        n = Point([0.0, 0.0, 0.0])
        t = None
        front_facing = True

        if hit:
            sqrtd = discriminant ** 0.5
            root = (-half_b - sqrtd) / a

            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    hit = False

        if hit:
            t=root
            p = ray.at(ray_origin, ray_direction, t)
            n = (p - center) / radius
            front_facing = is_front_facing(ray_direction, n)
            n = n if front_facing else -n

        return hit, t, p, n, front_facing

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        # hit_anything = False
        t_min = 0.0001
        closest_so_far = 9999999999.9
        hitRecord=empty_hit_record()
        hit_index = -1
        hit_tri_index = -1
        p = Point([0.0, 0.0, 0.0])
        n = Vector([0.0, 0.0, 0.0])
        front_facing = True
        i = 0

        # curr = self.bvh.bvh_root
        curr = self.bvh_root

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id,area_id = self.bvh.get_full_id(curr)
            # print("hit_aabb", obj_id)
            if obj_id != -1:

                cur_obj=self.objs_field[obj_id]
                hit, _t, _p, _n, _front_facing,_hit_tri_index = self.hit_obj(obj_id,
                                    ray_origin, ray_direction, t_min,
                                    closest_so_far)

                if hit:
                    closest_so_far = _t
                    hitRecord.is_hit=1
                    hitRecord.p=_p
                    hitRecord.normal=_n
                    hitRecord.t=_t
                    hitRecord.front_face=_front_facing
                    hitRecord.is_emmit=cur_obj.li_type
                    hitRecord.tri_idx=_hit_tri_index
                    hitRecord.hit_index=obj_id


                curr = next_id
            else:

                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):
                    # add left and right children

                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id

                else:
                    curr = next_id



        return hitRecord
        # return hit_anything, p, n, front_facing, hit_index,hit_tri_index

    @ti.func
    def get_mattype(self, obj_index ):
        return self.materials.get(obj_index)
    @ti.func
    def has_texture(self, obj_index ):
        rst=0
        obj = self.objs_field[obj_index]
        if obj.geo_type==1:
            rst=self.meshs_field.get_mesh(obj.local_idx).has_tex
        return rst
    @ti.func
    def brdf(self,hitRecord, wi, wo ):
        obj_index=hitRecord.hit_index
        obj = self.objs_field[obj_index]
        text_color = Color([0.0, 0.0, 0.0])
        has_texture=self.has_texture(obj_index)
        if has_texture==1:
            #暂时支持三角形网格的
            if obj.geo_type==1:
                text_color = self.getTextureColor(hitRecord)

        return self.materials.brdf(hitRecord,wi, wo ,has_texture,text_color)

    @ti.func
    def sample_triangle(self, obj_id, triangle_pos):
        obj = self.objs_field[obj_id]
        mesh = self.meshs_field.get_mesh(obj.local_idx)

        tri_idx=mesh.tris_start+triangle_pos
        triangl=self.meshs_field.get_triangle(tri_idx)
        v0,v1,v2=getVectors(triangl.vertices)
        x = ti.sqrt(ti.random())
        y = ti.random()
        coords = v0 * (1.0  - x) + v1 * (x * (1.0 - y)) + v2 * (x * y)

        _norm=Triangle.getNormal(mesh.normal_type,triangl, coords)

        normal = _norm
        return coords,normal

    @ti.func
    def hit_light(self,ray_origin,  out_dir,obj):

        t_min=0.001
        t_max=infinity
        outward_normal=Vector([0.0, 1.0, 0.0])
        coords=Vector([0.0, 0.0, 0.0])
        # t = (k - p.y) / out_dir.y
        is_hit=True
        if obj.geo_type==1:
            is_hit,t, coords, outward_normal, front_facing,hit_tri_index =self.hit_meshs(obj ,ray_origin, out_dir, t_min,t_max)


        return coords,outward_normal,is_hit


    @ti.func
    def pdf_light(self,   p, n, out_dir):
        pdf = 0.0

        if self.light_num[None]>0:
           weight = 1.0 / self.light_num[None]

           for k in range(self.n):

               obj = self.objs_field[k]
               if obj.li_type == 1:
                   coords, outward_normal, is_hit_light = self.hit_light(p,  out_dir,obj)
                   if is_hit_light:
                       wo_vec = coords - p
                       pdf += weight * wo_vec.norm_sqr() / (abs(wo_vec.normalized().dot(outward_normal)) * obj.area)
           #         hit, _p, _n, front_facing, index,hit_tri_index = self.hit_all(p, out_dir)
           #         hitobj = self.objs_field[index]
           #         if hit and hitobj.li_type==1 and hitobj.id==obj.id and front_facing:
           #             wo_vec = _p - p
           #             print(_p)
           #             pdf+=weight*wo_vec.norm_sqr()/(abs(wo_vec.normalized().dot(_n))*obj.area  )
        return pdf



    @ti.func
    def sample_light(self, ray_direction, hitRecord):
        #（多光源）随机一个光源的index
        # samplLightIdx = ti.cast(ti.random()*(self.light_num[None]-1),dtype=ti.i32)
        samplLightIdx = 0
        i=0
        coords=Point([0.0, 0.0, 0.0])
        normal=Point([0.0, 0.0, 0.0])
        emitted=Color([0.0, 0.0, 0.0])
        wi=Vector([0.0, 0.0, 0.0])

        pdf=0.0
        light_obj_id=-1.0
        if self.env_light !=0:

            le, wi, pdf=self.env_light.sample_light(hitRecord)
            if pdf>0:
                emitted=abs(le) /pdf
                # emitted=le*hitRecord.normal.dot(wi)/pdf


        if self.light_num[None]>0:
           for k in range(self.n):

              obj = self.objs_field[k]
              if obj.li_type ==1:
                  if i==samplLightIdx:
                      if obj.geo_type==1:
                        mesh = self.meshs_field.get_mesh(obj.local_idx)
                        pos, pdf=self.bvh.sample(mesh.bvh_root)
                        coords, normal=self.sample_triangle(k,pos)
                        isLi,le=self.materials.emitted(k)
                        if pdf>0.0:
                          sample_ray_dir = coords - hitRecord.p
                          wi = sample_ray_dir.normalized()
                          slHitRecord = self.hit_all(hitRecord.p, wi)
                          if slHitRecord.is_hit and (coords - slHitRecord.p).norm_sqr() < EPSILON:
                            emitted+=le*abs(hitRecord.normal.dot(wi)*slHitRecord.normal.dot(-wi))/sample_ray_dir.dot(sample_ray_dir)/pdf
                      break
                  else:
                      i+=1
        return coords, normal, pdf,emitted,wi
