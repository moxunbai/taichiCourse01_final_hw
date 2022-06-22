import taichi as ti
from render.vector import *
from render.meshtriangle import *
from time import time
from render.material import *
from render.scene import *
from render.camera import Camera
from render.Texture2D import *
from render.EnvLight import *
import math
import random
import sys
import json


@ti.data_oriented
class RenderEntry:

    def __init__(self, conf_fn):
        # switch to cpu if needed
        ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=int(time()), advanced_optimization=False)

        json_data = None
        with open(conf_fn, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
        if json_data is None:
            raise Exception('config  json is none! ')
        scene_param = json_data["scene"]
        ren_param = json_data["render"]
        self.output = ren_param["output"]
        film_param = scene_param["film"]
        cam_param = scene_param["camera"]

        aspect_ratio = film_param["aspect_ratio"]
        self.image_width = film_param["width"]
        self.image_height = int(self.image_width / aspect_ratio)
        self.samples_per_pixel = ren_param["spp"]
        self.max_depth = ren_param["max_depth"]

        self.img_writer = "ti"
        if "img_writer" in ren_param:
            self.img_writer = ren_param["img_writer"]

        vfrom = Point(cam_param["postion"])
        at = Point(cam_param["lookat"])
        up = Vector(cam_param["up"])
        focus_dist = cam_param["focus_dist"]
        aperture = cam_param["aperture"]
        fov = cam_param["fov"]
        self.cam = Camera(vfrom, at, up, fov, aspect_ratio, aperture, focus_dist)

        self.scene = Scene.GenScene(scene_param)
        # gen_scene(scene,scene_param)
        # scene.commit()

        self.film_pixels = ti.Vector.field(3, dtype=ti.f32)

        ti.root.dense(ti.ij,
                      (self.image_width, self.image_height)).place(self.film_pixels)

    @ti.func
    def ray_color(self, ray_org, ray_dir):

        col = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)

        coefficient = ti.Vector([1.0, 1.0, 1.0], dt=ti.f32)

        for i in range(self.max_depth):

            hitRecord = self.scene.hit_all(ray_org, ray_dir)
            if not hitRecord.is_hit:
                if self.scene.env_light != 0:
                    col = self.scene.env_light.get_radiance_bydir(ray_dir) * coefficient
                # print(col)
                else:
                    col = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                break
            else:

                if hitRecord.is_emmit:  # 光源
                    isLight, emittedCol = self.scene.materials.emitted(hitRecord.hit_index)

                    col = coefficient * emittedCol
                    break
                else:
                    mat_c, mat_type, mat_roughness, mat_ior = self.scene.get_mattype(hitRecord.hit_index)

                    if mat_type == 2:
                        sample_ok, wi = self.scene.materials.sample(ray_dir, hitRecord)
                        ray_org, ray_dir = hitRecord.p, wi.normalized()
                        _brdf = self.scene.brdf(hitRecord, ray_dir, ti.Vector([0.0, 0.0, 0.0], dt=ti.f32))
                        coefficient *= _brdf  # 衰减
                    else:
                        l_dir = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                        l_indir = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                        coords, normal, pdf_light, emitted, wi = self.scene.sample_light(ray_dir, hitRecord)
                        if pdf_light > 0.0:
                            l_dir = emitted * self.scene.brdf(hitRecord, ray_dir, wi)

                        sample_ok, wo = self.scene.materials.sample(ray_dir, hitRecord)
                        nolight_record = self.scene.hit_all(hitRecord.p, wo)
                        if not nolight_record.is_hit:
                            wo_pdf = self.scene.materials.sample_pdf(hitRecord, ray_dir, wo)
                            _brdf = self.scene.brdf(hitRecord, ray_dir, wo)
                            l_indir = self.scene.env_light.get_radiance_bydir(wo) * _brdf * hitRecord.normal.dot(
                                wo) / wo_pdf
                            col += l_dir + l_indir * coefficient

                            break
                        elif not nolight_record.is_emmit:

                            wo_pdf = self.scene.materials.sample_pdf(hitRecord, ray_dir, wo)
                            _brdf = self.scene.brdf(hitRecord, ray_dir, wo)
                            l_indir = _brdf * hitRecord.normal.dot(wo) / wo_pdf
                            ray_org, ray_dir = hitRecord.p, wo
                            # if hitRecord.hit_index == 4:
                            #    print(l_indir)

                        col += l_dir * coefficient
                        coefficient = l_indir * coefficient

                        if coefficient.norm() == 0:
                            break
        return col

    @ti.kernel
    def init_field(self):
        for i, j in self.film_pixels:
            self.film_pixels[i, j] = ti.Vector.zero(float, 3)

    @ti.kernel
    def cal_film_val(self):
        for i, j in self.film_pixels:
            val = self.film_pixels[i, j] / self.samples_per_pixel
            # film_pixels[i, j] =  val
            if self.img_writer == "ti":
                self.film_pixels[i, j] = clamp(ti.sqrt(val), 0.0, 0.999)
            else:
                self.film_pixels[i, j] = clamp(val, 0.0, 0.999)

    @ti.kernel
    def render_once(self):
        for i, j in self.film_pixels:
            (u, v) = ((i + ti.random()) / self.image_width, (j + ti.random()) / self.image_height)
            ray_org, ray_dir = self.cam.get_ray(u, v)
            ray_dir = ray_dir.normalized()
            self.film_pixels[i, j] += self.ray_color(ray_org, ray_dir)

    def commit(self):
        self.scene.commit()

    def run(self, out_fn=None):

        self.init_field()
        for k in range(int(self.samples_per_pixel)):
            self.render_once()
        self.cal_film_val()

        if out_fn is None:
            out_fn = self.output
        # cv2.imwrite(out_fn, self.film_pixels.to_numpy()  )
        if self.img_writer == "ti":
            ti.tools.imwrite(self.film_pixels.to_numpy(), out_fn)
        else:
            img_data = self.film_pixels.to_numpy()
            # cv保存图片的和ti的效果有差异,一是RGB颜色值顺序是反的，图像朝向也反了90度
            # mid=img_data[:,:,2]
            # img_data[:,:,2]=img_data[:,:,0]
            # img_data[:,:,0]=mid
            cv2.imwrite(out_fn, img_data * 255)
