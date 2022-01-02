# import taichi_glsl as ts
import taichi as ti
import math

vec3f = ti.types.vector(3, ti.f32)
vec3i = ti.types.vector(3, ti.i32)
# Vector = ts.vec3
# Color = ts.vec3
# Point = ts.vec3

Vector = vec3f
Color = vec3f
Point = vec3f

WHITE = Color(1.0, 1.0, 1.0)
BLUE = Color(0.5, 0.7, 1.0)
RED = Color(1.0, 0.0, 0.0)

infinity = float("inf")

@ti.func
def random_in_unit_disk():
    theta = ti.random() * math.pi * 2.0
    r = ti.random()**0.5

    return Vector(r * ti.cos(theta), r * ti.sin(theta), 0.0)


@ti.func
def random_in_hemisphere(normal):
    vec = random_in_unit_sphere()
    if vec.dot(normal) < 0:
        vec = -vec
    return vec


@ti.func
# def random_in_unit_sphere():
#     theta = ti.random() * math.pi * 2.0
#     v = ti.random()
#     phi = ti.acos(2.0 * v - 1.0)
#     r = ti.random()**(1 / 3)
#     return Vector(r * ti.sin(phi) * ti.cos(theta),
#                   r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi))
@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()


@ti.func
def clamp(v,minv,maxv):
    x= maxv if v.x>maxv else (minv if v.x<minv else v.x)
    y= maxv if v.y>maxv else (minv if v.y<minv else v.y)
    z= maxv if v.z>maxv else (minv if v.z<minv else v.z)
    return ti.Vector([x, y, z])

@ti.func
def get_sphere_coordinate(n):
    theta =ti.acos(n[1])
    phi = ti.atan2(n[0], n[2])
    if phi <0.0:
        phi+=  (2*math.pi)
    u=phi/math.pi/2
    v=1-theta/math.pi
    # u = (ti.atan2(-n[0], -n[2]) / math.pi  + 1) / 2
    # v = 1 - ti.acos(n[1]) /math.pi
    return ti.Vector([u, v])


def makeTransformations(scale,rota_angle,trans):
   scale_mat = ti.Matrix([[scale, 0, 0, 0],
                          [0, scale, 0, 0],
                          [0, 0, scale, 0],
                          [0, 0, 0, 1]])
   #绕y轴旋转
   rota_mat = ti.Matrix([[ti.cos(rota_angle), 0, ti.sin(rota_angle), 0],
                         [0, 1, 0, 0],
                         [-ti.sin(rota_angle), 0, ti.cos(rota_angle), 0],
                         [0, 0, 0, 1]])
   trans_mat2 = ti.Matrix([[1, 0, 0, trans[0]],
                           [0, 1, 0, trans[1]],
                           [0, 0, 1, trans[2]],
                           [0, 0, 0, 1]])
   return trans_mat2@rota_mat@scale_mat
