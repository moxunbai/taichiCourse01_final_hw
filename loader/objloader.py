import numpy as np
import taichi as ti
# import pygame, OpenGL

'''
the original is here https://www.pygame.org/wiki/OBJFileLoader
@2018-1-2 author chj
change for easy use
'''


class OBJ:
    def __init__(self,  filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl = None

        # self.face = ti.Vector.field(3, dtype=ti.f32)
        # self.direction = ti.Vector.field(3, dtype=ti.f32)

        material = None
        for line in open( filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]

                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                # print(values[1])
                # self.mtl = MTL(fdir,values[1])
                self.mtl = [ values[1]]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    # print("w len:",len(w))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def create_bbox(self):
        # self.vertices is not None
        ps = np.array(self.vertices)
        vmin = ps.min(axis=0)
        vmax = ps.max(axis=0)

        self.bbox_center = (vmax + vmin) / 2
        self.bbox_half_r = np.max(vmax - vmin) / 2

    @staticmethod
    def write_obj(fn, vertices, faces, vnormals=None, texcoords=None):
        with open(fn, 'w') as file_object:
            nv = len(vertices)
            nf = len(faces)
            if nv > 0:
                file_object.write("# Object spot_triangulated_good.obj\n")
                file_object.write("# Vertices:" + str(nv) + "\n")
                file_object.write("# Faces:" + str(nf) + "\n")
                for i in range(nv):

                    if vnormals is not None and len(vnormals) > 0:
                        nor = vnormals[i]
                        file_object.write("vn " + str(nor[0]) + " " + str(nor[1]) + " " + str(nor[2]) + "\n")
                    if texcoords is not None and len(texcoords) > 0:
                        tex = texcoords[i]
                        file_object.write("vt " + str(tex[0]) + " " + str(tex[1]) + "\n")
                    vet = vertices[i]
                    file_object.write("v " + str(vet[0]) + " " + str(vet[1]) + " " + str(vet[2]) + "\n")

                file_object.write("\n")
                for i in range(nf):
                    face = faces[i]
                    txt = [[str(face[0])], [(face[1])], [str(face[2])]]
                    vn=[0,0,0]
                    vtex=[0,0,0]
                    if vnormals is not None and len(vnormals) > 0:
                        vn=face
                        # txt = [txt[0] + "/" + str(face[0]), txt[1] + "/" + str(face[1]), txt[2] + "/" + str(face[2])]
                    txt[0].append(vn[0])
                    txt[1].append(vn[1])
                    txt[2].append(vn[2])
                    if texcoords is not None and len(texcoords) > 0:
                        vtex=face
                        # txt = [txt[0] + "/" + str(face[0]), txt[1] + "/" + str(face[1]),
                        #        txt[2] + "/" + str(face[2])]
                    txt[0].append(vtex[0])
                    txt[1].append(vtex[1])
                    txt[2].append(vtex[2])

                    txt[0]="/".join(txt[0])
                    txt[1]="/".join(txt[1])
                    txt[2]="/".join(txt[2])
                    file_object.write("f " + " ".join(txt) + "\n")
                file_object.write("# End of File \n")

