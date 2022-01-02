import taichi as ti
import cv2
import numpy as np
from .vector import *
from render.Texture2D  import *

def val_illumination(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def alias_method(hdr_img_data):
    env_h = hdr_img_data.shape[0]
    env_w = hdr_img_data.shape[1]
    env_light_illumination_sum = 0.0
    env_light_pdf = np.zeros((env_h, env_w), dtype=np.float32)
    env_light_alias_table_p = np.zeros((env_h * env_w, 2), dtype=np.float32)
    env_light_alias_table_idx = np.zeros((env_h * env_w, 2, 2), dtype=np.int)
    queue_p_greater_than_one = []
    queue_p_leq_one = []
    for i in range(env_h):
        for j in range(env_w):
            env_light_illumination_sum += val_illumination(hdr_img_data[i, j])

    for i in range(env_h):
        for j in range(env_w):
            p = float(val_illumination(hdr_img_data[i, j]) * env_h * env_w) / env_light_illumination_sum
            env_light_pdf[i, j] = p

            if p > 1:
                queue_p_greater_than_one.append((p, (i, j)))
            else:
                queue_p_leq_one.append((p, (i, j)))
    while (len(queue_p_greater_than_one) > 0 and len(queue_p_leq_one) > 0):
        p_leq_one_info = queue_p_leq_one.pop()

        idx = p_leq_one_info[1]
        p = p_leq_one_info[0]
        if abs(p - 1.0) < 1e-5:
            env_light_alias_table_p[idx[0] * env_w + idx[1]] = (1.0, 0.0)
            env_light_alias_table_idx[idx[0] * env_w + idx[1]] = (idx, (-1, -1))
        else:
            p_gt_one_info = queue_p_greater_than_one.pop()
            p2 = p_gt_one_info[0]

            idx2 = p_gt_one_info[1]

            env_light_alias_table_p[idx[0] * env_w + idx[1]] = (p, 1 - p)
            env_light_alias_table_idx[idx[0] * env_w + idx[1]] = (idx, idx2)

            if (p2 - (1 - p) > 1):
                queue_p_greater_than_one.append((p2 - (1 - p), idx2))
            else:
                queue_p_leq_one.append((p2 - (1 - p), idx2))

    while len(queue_p_leq_one) > 0:
        p_leq_one_info = queue_p_leq_one.pop()
        idx = p_leq_one_info[1]
        env_light_alias_table_p[idx[0] * env_w + idx[1]] = (1.0, 0.0)
        env_light_alias_table_idx[idx[0] * env_w + idx[1]] = [idx, [-1, -1]]

    while len(queue_p_greater_than_one) > 0:
        p_gt_one_info = queue_p_greater_than_one.pop()

        idx = p_gt_one_info[1]
        env_light_alias_table_p[idx[0] * env_w + idx[1]] = [1.0, 0.0]
        env_light_alias_table_idx[idx[0] * env_w + idx[1]] = [idx, (-1, -1)]
    return     env_light_pdf,env_light_alias_table_p,env_light_alias_table_idx

def rgbe2float(rgbe):
    res = np.zeros((rgbe.shape[0], rgbe.shape[1], 3))
    p = rgbe[:, :, 3] > 0
    m = 2.0 ** (rgbe[:, :, 3][p] - 136.0)
    res[:, :, 0][p] = rgbe[:, :, 0][p] * m
    res[:, :, 1][p] = rgbe[:, :, 1][p] * m
    res[:, :, 2][p] = rgbe[:, :, 2][p] * m
    return res


def readHdr(fileName):
    fileinfo = {}
    with open(fileName, 'rb') as fd:

        tline = fd.readline().strip().decode()

        if len(tline) < 3 or tline[:2] != '#?':
            print('invalid header')
            return
        fileinfo['identifier'] = tline[2:]

        tline = fd.readline().strip().decode()
        while tline:
            n = tline.find('=')
            if n > 0:
                fileinfo[tline[:n].strip()] = tline[n + 1:].strip()
            tline = fd.readline().strip().decode()

        tline = fd.readline().strip().decode().split(' ')
        fileinfo['Ysign'] = tline[0][0]
        fileinfo['height'] = int(tline[1])
        fileinfo['Xsign'] = tline[2][0]
        fileinfo['width'] = int(tline[3])

        # data = [ord(d) for d in fd.read()]
        data = [ d  for d in fd.read()]
        height, width = fileinfo['height'], fileinfo['width']
        if width < 8 or width > 32767:
            data.resize((height, width, 4))
            return rgbe2float(data)

        img = np.zeros((height,width,  4))
        dp = 0
        for h in range(height):
            if data[dp] != 2 or data[dp + 1] != 2:
                print('this file is not run length encoded')
                print(data[dp:dp + 4])
                return
            if data[dp + 2] * 256 + data[dp + 3] != width:
                print('wrong scanline width')
                return
            dp += 4
            for i in range(4):
                ptr = 0
                while (ptr < width):
                    if data[dp] > 128:
                        count = data[dp] - 128
                        if count == 0 or count > width - ptr:
                            print('bad scanline data')
                        img[ h,ptr:ptr + count, i] = data[dp + 1]
                        ptr += count
                        dp += 2
                    else:
                        count = data[dp]
                        dp += 1
                        if count == 0 or count > width - ptr:
                            print('bad scanline data')
                        # img[h, ptr:ptr + count, i] = data[dp: dp + count]
                        img[ h,ptr:ptr + count, i] = data[dp: dp + count]
                        ptr += count
                        dp += count
        rst=rgbe2float(img)
        # m1, m2 = rst.max(), rst.min()
        # rst = (rst - m2) / (m1 - m2)
        # rst1 = rst[:, :, 0].copy()
        # rst[:, :, 0] = rst[:, :, 2]
        # rst[:, :, 2] = rst1
        return rst


@ti.data_oriented
class EnvLight:
    def __init__(self,intensity=1.0,  color=[0.0,0.0,0.0], fn=None):
        self.intensity=ti.field(ti.f32,shape=())
        self.intensity[None] =intensity
        self.color=ti.Vector(color)
        self.env_width = 0
        self.env_height = 0
        self.has_texture=fn is not None

        self.texture_data = ti.Vector.field(3, dtype=ti.f32)
        self.env_light_alias_table_p = ti.Vector.field(2, dtype=ti.f32)
        self.env_light_alias_table_idx = ti.Matrix.field(2, 2, dtype=ti.i32)
        self.env_light_pdf = ti.field(dtype=ti.f32)
        if self.has_texture:

           img_data = readHdr(fn)
           # img_data1  = Texture2D(fn,reshape=False )
           # img_data = img_data1.data

           # img_data=img_data.reshape(img_data.shape[1],img_data.shape[0],img_data.shape[2])
           # print(img_data.shape)

           self.env_width = img_data.shape[0]
           self.env_height = img_data.shape[1]


           env_light_pdf, env_light_alias_table_p, env_light_alias_table_idx=alias_method(img_data)

           ti.root.dense(ti.ij, (self.env_width,self.env_height)).place(self.texture_data,self.env_light_pdf)
           ti.root.dense(ti.i, (self.env_width*self.env_height)).place(self.env_light_alias_table_p,self.env_light_alias_table_idx)



           self.texture_data.from_numpy(img_data)
           # self.env_light_pdf.from_numpy(env_light_pdf)
           # self.env_light_alias_table_idx.from_numpy(env_light_alias_table_idx)
           # self.env_light_alias_table_p.from_numpy(env_light_alias_table_p)
           # self.init_data(img_data)
           del img_data
        else:
            ti.root.dense(ti.ij, (1)).place(self.texture_data, self.env_light_pdf)
            ti.root.dense(ti.i, (1)).place(self.env_light_alias_table_p,
                                                                          self.env_light_alias_table_idx)

    @ti.func
    def get_radiance(self,uv):
        return self.intensity[None]*self.color*self.sample_nearest(uv)
    @ti.kernel
    def init_data(self,tex_data: ti.ext_arr()):
        for i,j in self.texture_data:
            self.texture_data[i,j]=[tex_data[i,j,0],tex_data[i,j,1],tex_data[i,j,2]]
    @ti.func
    def get_radiance_bydir(self,dir):
        rst=ti.Vector([0.0,0.0,0.0])
        if self.has_texture:
          uv=get_sphere_coordinate(dir.normalized())
          rst =clamp(self.get_radiance(uv), 0.0, 0.9999)
        else:
          rst =self.intensity[None]*self.color
        return rst

    @ti.func
    def sample_nearest(self, uv):
       # u =  1-uv[0]
       # v = uv[1]
       u =  1-uv[1]
       v = 1-uv[0]
       xf = self.env_width * u-0.5
       yf = self.env_height * v-0.5

       x = ti.cast(xf,ti.i32)
       y = ti.cast(yf,ti.i32)
       return self.texture_data[x, y]

    @ti.func
    def sample_light(self,hitRecord ):
        rand_n = ti.cast(ti.random() * (self.env_height * self.env_width - 1),ti.i32)
        rand_p = ti.random()
        sample_idx =ti.Vector([0,0])
        sa_a_idx=self.env_light_alias_table_idx[rand_n]
        if rand_p <= self.env_light_alias_table_p[rand_n][0] :
            sample_idx = ti.Vector([sa_a_idx[0,0],sa_a_idx[0,1]])
        else:
            sample_idx = ti.Vector([sa_a_idx[1,0],sa_a_idx[1,1]])

        i = sample_idx[0]
        j = sample_idx[1]

        v = i /self.env_height
        u = j/self.env_width
        theta = math.pi * (1 - v)
        phi = 2 * math.pi * u
        wi = ti.Vector([ti.sin(theta) * ti.sin(phi), ti.cos(theta), ti.sin(theta) * ti.cos(phi)]).normalized()
        Le = self.get_radiance(ti.Vector([u, v]))

        sin = wi.cross(hitRecord.normal).norm()
        pd_env = 1.0
        if ( abs(sin - 0.0) > 1e-4):
            if self.env_light_pdf[i,j]>0.0:
                pd_env = min(self.env_light_pdf[i,j] / abs(sin) / (2 * math.pi * math.pi), 1.0)
        return Le,wi,pd_env
    # @ti.func
    # def tex_sample(self,uv):
    #     return self.texture_data
