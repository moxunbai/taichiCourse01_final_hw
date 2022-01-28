import os
import urllib.request
import pika
import sys
import json
# import sim_paper_plan
from engine.sim_entry import SimEntry
import render_paper_plan_consumer as rconsumer
import taichi as ti
import _thread

model_desc_template={
        "type":"MeshTriangle",
        "key":"paper_plan",
        "filename":"",
        "material":{
          "type":"Lambert",
          "color":[1.0,1.0,1.0]
        },
        "transformation":{
          "translation":[ 4260,1800,-150],
          "scale":18,
          "routeY":0.0
        }
      }

if __name__ == '__main__':
    base_config = None
    with open("./config/task_pplan.json", 'r', encoding='utf8')as fp:
        base_config = json.load(fp)
    mq_conf=base_config["mq"]
    io_conf=base_config["io"]
    out_base_dir=io_conf["out_base_dir"]
    task_scope = base_config["task_scope"]
    if not os.path.exists(out_base_dir):
        os.makedirs(out_base_dir)
    sim_iter_num=base_config["sim_iter_num"]
    sim_config_fn=base_config["sim_config_base"]
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=mq_conf["host_ip"]))
    channel = connection.channel()
    render_queue="queue_for_render"
    ren_succ_counter_que = "queue_for_render_count"
    channel.queue_declare(queue=ren_succ_counter_que)
    channel.queue_declare(queue=render_queue)

    sim=SimEntry(sim_config_fn)


    def after_sim_frame(out_dir,frame_id, obj_fn):
        msg={"server_host":base_config["http_server_host"]}
        tmp_model=model_desc_template.copy()
        tmp_model["filename"]=out_dir+"/"+obj_fn[0]
        msg["model"]=tmp_model
        msg["obj_fn"]=obj_fn[0]
        msg["obj_dir"]=out_dir
        msg["frame_id"]=frame_id

        channel.basic_publish(exchange='',
                              routing_key=render_queue,
                              body=json.dumps(msg, indent=2))
    if task_scope=="master":
       print("sim start")
       # sim_paper_plan.run(sim_iter_num,out_base_dir, after_sim_frame )
       sim.run(out_base_dir, after_sim_frame )
       print("sim over")

    ##重置taichi 然后进行渲染
    ti.reset()
    print("render start")
    rconsumer.render_by_mq()
    print("render over")



