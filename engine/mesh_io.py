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
    # with open(fn, 'w') as file_object:
        nv = len(vertices)
        nf = len(faces)
        if nv > 0:
            fo = open(fn, "w")
            print("# Object spot_triangulated_good.obj\n", file = fo)
            print("# Vertices:%d\n"%(nv), file = fo)
            print("# Faces::%d\n"%(nf), file = fo)
            has_tex = texcoords is not None and len(texcoords) > 0
            has_normal=vnormals is not None and len(vnormals) > 0
            for i in range(nv):

                if has_normal:
                    print("vn %f %f %f" % (vnormals[i, 0], vnormals[i, 1], vnormals[i, 2]), file=fo)

                if has_tex:
                    print("vt %f %f" % (texcoords[i, 0], texcoords[i, 1] ), file=fo)
                vet = vertices[i]
                print("v %f %f %f" % (vet[0], vet[1], vet[2]), file=fo)

            for i in range(nf):
                # face =[str(x) for x in faces[i]]
                face = faces[i]
                txt = [[face[0]], [face[1]], [face[2]]]
                vn = [0, 0, 0]
                vtex = [0, 0, 0]
                if has_tex:
                    vtex=face
                if has_normal:
                    vn=face
                print("f %d/%d/%d %d/%d/%d %d/%d/%d" % (face[0],vtex[0],vn[0],face[1],vtex[1],vn[1],face[2],vtex[2],vn[2]), file=fo)

            print("# End of File \n", file=fo)
            fo.close()


def zoom_model(points,scale,div=1):

    bound_min=np.amin(points,0 )
    bound_max=np.amax(points,0 )
    center=(bound_min+bound_max)/2
    result=np.subtract(points,center)*scale
    return np.add(result,center/div)
    # for i in len(points):
