import taichi as ti
from .vector import *
import render.ray
from .material import Materials
import random
import numpy as np
# from bvh import BVH

HitRecord = ti.types.struct(is_hit=ti.i32,p=Point, normal=Vector, t=ti.f32, front_face=ti.i32, is_emmit=ti.i32, tri_idx=ti.i32, hit_index=ti.i32)
@ti.func
def empty_hit_record():
    ''' Constructs an empty hit record'''
    return HitRecord(is_hit=0,p=Point(0.0), normal=Vector(0.0), t=0.0, front_face=1,is_emmit=0, tri_idx=-1, hit_index=-1)

@ti.func
def is_front_facing(ray_direction, normal):
    return ray_direction.dot(normal) < 0.0

class Sphere:
    def __init__(self, center, radius, material,key=None):
        self.center = center
        self.radius = radius
        self.material = material
        self.area=4*math.pi*radius*radius
        self.id = -1
        self.key = key
        self.box_min = [
            self.center[0] - radius, self.center[1] - radius,
            self.center[2] - radius
        ]
        self.box_max = [
            self.center[0] + radius, self.center[1] + radius,
            self.center[2] + radius
        ]

    @property
    def bounding_box(self):
        return self.box_min, self.box_max


BRANCH = 1.0
LEAF = 0.0

class Spheres:
    def __init__(self, spheres):
        sphe_type = ti.types.struct(
            center=vec3f, radius=ti.f32
        )
        nSph = len(spheres)
        self.n = nSph
        self.datas=spheres
        self.spheres = sphe_type.field(shape=(nSph if nSph>0 else 1))


    def setup_data_cpu(self):
        for i in range(self.n):
            self.spheres[i].center = self.datas[i].center
            self.spheres[i].radius = self.datas[i].radius
    @ti.func
    def get(self, i ):
        return self.spheres[i]
