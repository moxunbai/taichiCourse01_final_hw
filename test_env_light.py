import taichi as ti
from render.vector import *
from render.meshtriangle  import *
from time import time
from render.material import *
from render.scene import *
from render.camera import Camera
from render.Texture2D  import *
from render.EnvLight  import *
import math
import random

# switch to cpu if needed
ti.init(arch=ti.gpu,random_seed=int( time()) )
# ti.set_logging_level(ti.DEBUG)

if __name__ == '__main__':

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

   aspect_ratio = 0.5
   image_width = 700
   image_height = int(image_width / aspect_ratio)
   # rays = ray.Rays(image_width, image_height)
   film_pixels = ti.Vector.field(3, dtype=ti.f32)

   ti.root.dense(ti.ij,
                 (image_width, image_height)).place(film_pixels )
   samples_per_pixel = 500
   max_depth = 10

   red = Lambert([0.65, .05, .05])
   white = Lambert([.73, .73, .73])
   white1 = Lambert([1.0, 1.0, 1.0])
   green = Lambert([.12, .45, .15])
   light = Lambert_light([15, 15, 15])

   metal = Metal([0.7, 0.6, 0.5], 0.1)
   glass = Dielectric(1.5)

   moveVec=[485,0,245]
   trans= Translate( makeTransformations(180,math.pi/6,moveVec))
   spot = MeshTriangle("./models/spot/spot_triangulated_good.obj", white ,trans)
   spot_tex=Texture2D("./models/spot/spot_texture.png")
   spot.set_texture(spot_tex)
   # spot = MeshTriangle("./models/spot/spot_falling_599.obj", white ,None,"./models/spot/spot_texture.png")
   # spot = MeshTriangle("./models/spot/spot_falling_599.obj", white ,None )

   moveVec2 = [155, 25, 125]
   trans2 = Translate(makeTransformations(910, math.pi, moveVec2))
   # bunny = MeshTriangle("./models/bunny/bunny.obj", white,trans2)

   left = MeshTriangle("./models/cornellbox/left.obj", red)
   right = MeshTriangle("./models/cornellbox/right.obj", green)
   floor = MeshTriangle("./models/cornellbox/floor.obj", white)
   light_ = MeshTriangle("./models/cornellbox/light.obj", light)

   shortbox  = MeshTriangle("./models/cornellbox/shortbox.obj", white1)
   # tallbox  = MeshTriangle("./models/cornellbox/tallbox.obj", white)

   # world

   scene = Scene()
   env_light = EnvLight(3.0,Color([1.0, 1.0, 1.0]),"./data/je_gray_park_4k.hdr")
   scene.set_env_light(env_light)
   scene.add(Sphere([610.0, 310.0, 290.0], 90.0, glass))
   scene.add(Sphere([10.0, 510.0, 290.0], 90.0, metal))

   # scene.add(Sphere([370.0, 310.0, 390.0], 90.0, white))

   # scene.add(light_, 1)
   # scene.add(floor)
   # scene.add(left)
   scene.add(spot)
   # scene.add(right)
   # scene.add(bunny)

   # scene.add(tallbox)
   scene.add(shortbox)
   scene.commit()

   # camera
   vfrom = Point([278.0, 273.0, -800.0])
   at = Point([278.0, 273.0, 0.0])
   up = Vector([-1.0, 0.0, 0.0])
   focus_dist =  10.0
   aperture = 0.0
   cam = Camera(vfrom, at, up, 90.0, aspect_ratio, aperture, focus_dist)



   @ti.func
   def ray_color(ray_org, ray_dir):

      col = ti.Vector([0.0, 0.0, 0.0])

      coefficient = ti.Vector([1.0, 1.0, 1.0])

      for i in range(max_depth):

         hitRecord = scene.hit_all(ray_org, ray_dir)

         if not hitRecord.is_hit:
            if  scene.env_light !=0:
              col=scene.env_light.get_radiance_bydir(ray_dir)
            # print(col)
            break
         else:
            # if i == 0:
            #   print(hitRecord.hit_index)
            if hitRecord.is_emmit:  # 光源
               isLight, emittedCol = scene.materials.emitted(hitRecord.hit_index)
               # if hitRecord.front_face:
               col = coefficient * emittedCol
               break
            else:
               mat_c, mat_type, mat_roughness, mat_ior =scene.get_mattype(hitRecord.hit_index)

               if mat_type == 2:
                  sample_ok, wi = scene.materials.sample(ray_dir, hitRecord)
                  ray_org, ray_dir = hitRecord.p, wi.normalized()
                  # coefficient *= attenuation  # 衰减
               else:
                   l_dir = ti.Vector([0.0, 0.0, 0.0])
                   l_indir = ti.Vector([0.0, 0.0, 0.0])
                   coords, normal, pdf_light, emitted,wi = scene.sample_light(ray_dir, hitRecord)
                   if pdf_light>0.0:
                      l_dir =emitted*scene.brdf(hitRecord ,ray_dir,wi )
                      # if l_dir.norm_sqr()>0:
                      #   print(l_dir)
                      # sample_ray_dir = coords - hitRecord.p
                      # s_r_dir_n = sample_ray_dir.normalized()
                      # slHitRecord = scene.hit_all(hitRecord.p, s_r_dir_n)
                      #
                      # if slHitRecord.is_hit and (coords - slHitRecord.p).norm_sqr() < EPSILON:
                      #    l_dir = emitted *scene.brdf(hitRecord ,ray_dir,s_r_dir_n )*\
                      #            abs(hitRecord.normal.dot(s_r_dir_n)*slHitRecord.normal.dot(-s_r_dir_n))/sample_ray_dir.dot(sample_ray_dir)/pdf_light

                   sample_ok,wo = scene.materials.sample(ray_dir,hitRecord)
                   nolight_record=scene.hit_all(hitRecord.p, wo)
                   if not nolight_record.is_hit:
                       wo_pdf = scene.materials.sample_pdf(hitRecord, ray_dir, wo)
                       _brdf = scene.brdf(hitRecord, ray_dir, wo)
                       l_indir=scene.env_light.get_radiance_bydir(wo) * _brdf * hitRecord.normal.dot(wo) / wo_pdf
                       col +=l_dir+ l_indir * coefficient
                       break
                   elif  not nolight_record.is_emmit:

                      wo_pdf = scene.materials.sample_pdf(hitRecord,ray_dir, wo)
                      _brdf=scene.brdf(hitRecord , ray_dir, wo )
                      l_indir=_brdf*hitRecord.normal.dot(wo)/wo_pdf
                      ray_org, ray_dir=hitRecord.p, wo
                      # if hitRecord.hit_index == 4:
                      #    print(l_indir)

                   col +=l_dir*coefficient
                   coefficient  = coefficient *l_indir

                   if coefficient.norm()==0:
                      break


      return col


   @ti.kernel
   def init_field():
      for i, j in film_pixels:
         film_pixels[i, j] = ti.Vector.zero(float, 3)
   @ti.kernel
   def cal_film_val():
      for i, j in film_pixels:
         val = film_pixels[i, j] / samples_per_pixel
         # film_pixels[i, j] =  val
         # film_pixels[i, j] = clamp(ti.sqrt(val), 0.0, 0.999)
         film_pixels[i, j] = clamp(val, 0.0, 0.999)
   @ti.kernel
   def render_once():
      for i, j in film_pixels:
         (u, v) = ((i + ti.random()) / image_width, (j + ti.random()) / image_height)
         ray_org, ray_dir = cam.get_ray(u, v)
         ray_dir = ray_dir.normalized()
         film_pixels[i, j] += ray_color(ray_org, ray_dir)


   t = time()
   print('starting rendering')
   init_field()
   for k in range(int(samples_per_pixel)):
      render_once()
   cal_film_val()
   print(time() - t)
   cv2.imwrite('out.jpg', film_pixels.to_numpy() * 255)
   # ti.imwrite(film_pixels.to_numpy(), 'out.png')
