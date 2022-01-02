from plyfile import PlyData
import numpy as np


def load_mesh(fn, scale=1, offset=(0, 0, 0)):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(elements['vertex_indices']):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    return triangles


def write_point_cloud(fn, pos_and_color):
    num_particles = len(pos_and_color)
    with open(fn, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar placeholder
end_header
"""
        f.write(str.encode(header))
        f.write(pos_and_color.tobytes())


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
                face =[str(x) for x in faces[i]]
                txt = [[face[0]], [face[1]], [face[2]]]
                vn = ["0", "0", "0"]
                vtex = ["0", "0", "0"]

                if texcoords is not None and len(texcoords) > 0:
                    vtex = face

                txt[0].append(vtex[0])
                txt[1].append(vtex[1])
                txt[2].append(vtex[2])

                if vnormals is not None and len(vnormals) > 0:
                    vn = face
                txt[0].append(vn[0])
                txt[1].append(vn[1])
                txt[2].append(vn[2])

                txt[0] = "/".join(txt[0])
                txt[1] = "/".join(txt[1])
                txt[2] = "/".join(txt[2])
                file_object.write("f " + " ".join(txt) + "\n")
            file_object.write("# End of File \n")


def transform(points,scale):

    bound_min=np.amin(points,0 )
    bound_max=np.amax(points,0 )
    center=(bound_min+bound_max)/2
    result=np.subtract(points,center)*scale
    return np.add(result,center/6)
    # for i in len(points):
