import os
from render.render_entry  import *



if __name__ == '__main__':
   conf_fn=None
   if len(sys.argv)==1:
      conf_fn ="re_conf_cornellbox.json"
   else:
      conf_fn = sys.argv[1]
   if not os.path.exists("out"):
       os.makedirs("out")


   entry=RenderEntry(conf_fn)
   scene=entry.scene
   entry.commit()
   entry.run()

   # templ={
   #      "type":"MeshTriangle",
   #      "key":"paper_plan",
   #      "filename":"./data/models/paper_plan/00000.obj",
   #      "material":{
   #        "type":"Lambert",
   #        "color":[1.0,1.0,1.0]
   #      },
   #      "transformation":{
   #        "translation":[600,800,323],
   #        "scale":1,
   #        "routeY":0.0
   #      }
   #    }
   #
   # obj_fns=["00000.obj","00001.obj","00002.obj","00020.obj","00036.obj","00050.obj","00060.obj","00062.obj","00076.obj"]
   # str_mod = []
   # for fn  in obj_fns:
   #     pre=templ.copy()
   #     pre["filename"]="./data/models/paper_plan/"+fn
   #     str_mod.append(pre)
   #
   # for i in range(len(str_mod)):
   #    if len(scene.objects)==0:
   #        scene.add(Scene.GenMeshObj(str_mod[i]))
   #        entry.commit()
   #    else:
   #        obj=Scene.GenMeshObj(str_mod[i])
   #        scene.update_obj(obj)
   #
   #    entry.run("sim_pplan_{0}.jpg".format(str(i)))