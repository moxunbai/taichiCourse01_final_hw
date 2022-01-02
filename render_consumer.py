import os
import urllib.request
import pika
import sys
import json
from render.render_entry  import *


def render_by_mq():
    base_config = None
    with open("./config/base.json", 'r', encoding='utf8')as fp:
        base_config = json.load(fp)
    mq_conf = base_config["mq"]
    io_conf = base_config["io"]
    http_server_host = base_config["http_server_host"]
    out_base_dir = io_conf["out_base_dir"]
    img_dir = io_conf["img_dir"]
    input_source = io_conf["input_source"]
    img_fn = io_conf["img_fn"]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    tmp_dir=None
    if input_source=="remote":
        tmp_dir=out_base_dir+"/tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    #创建
    entry = RenderEntry(base_config["render_base"])
    scene = entry.scene

    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=mq_conf["host_ip"]))
    channel = connection.channel()
    render_queue = "queue_for_render"
    ren_succ_counter_que="queue_for_render_count"
    channel.queue_declare(queue=ren_succ_counter_que)

    def on_message(ch, method, propertities, body):

        msg_json = json.loads(body)
        frame_id=msg_json["frame_id"]
        print("rendering frame_id:",frame_id)
        modeljson=msg_json["model"]
        frame_id=msg_json["frame_id"]
        if input_source == "remote":
            ser_host=msg_json["server_host"]
            tmp_file=tmp_dir+"/"+msg_json["obj_fn"]
            urllib.request.urlretrieve(ser_host+"/"+msg_json["obj_dir"]+"/"+msg_json["obj_fn"], tmp_file)
            modeljson["filename"]=tmp_file


        if len(scene.objects) == 0:
            scene.add(Scene.GenMeshObj(modeljson))
            entry.commit()
        else:
            obj = Scene.GenMeshObj(modeljson)
            scene.update_obj(obj)
        rendered_img_fn=(img_dir+img_fn).format(str(frame_id))
        entry.run(rendered_img_fn)

        msg_for_frame={"frame_id":"frame_id","img_fn":rendered_img_fn}
        msg_for_frame["server_host"] = http_server_host

        ch.basic_publish(exchange='',
                              routing_key=ren_succ_counter_que,
                              body=json.dumps(msg_for_frame, indent=2))
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(render_queue,  # 队列名
                          on_message )
    channel.start_consuming()
    print("after start consum")

