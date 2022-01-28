import os
import urllib.request
import pika
import sys
import json
import cv2

class ImgDoneListener():
    def __init__(self,channel,qu_name,max_num,config):
        self.complete_count=0
        self.ch=channel
        self.queue=qu_name
        self.max_num=max_num
        self.config=config

        pass

    def make_vedio(self):
        io_json=self.config["io"]
        vedio_json=self.config["vedio"]
        img_dir=io_json["img_dir"]
        img_fn=io_json["img_fn"]
        vedio_dir=io_json["vedio_dir"]
        vedio_fn=io_json["vedio_fn"]
        if not os.path.exists(vedio_dir):
            os.makedirs(vedio_dir)

        size=vedio_json["size"]
        size=(size[0],size[1])
        frames=vedio_json["frames"]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowrite = cv2.VideoWriter(vedio_dir+"/"+vedio_fn, fourcc, frames, size)  # 60是帧数，size是图片尺寸
        img_array = []

        for i in range(self.max_num):
            filename = (img_dir+ img_fn).format(i)
            img = cv2.imread(filename)
            if img is None:
                print(filename + " is error!")
                continue
            img_array.append(img)


        for i in range(len(img_array)):
            videowrite.write(img_array[i])
    def on_message(self,ch, method, propertities, body):
        msg_json = json.loads(body)
        io_json = self.config["io"]
        input_source = io_conf["input_source"]
        msg_ser=msg_json["server_host"]
        local_ser=self.config["http_server_host"]
        is_remote = input_source == "remote"
        if local_ser!=msg_ser:
            url =msg_ser+ msg_json["img_fn"]
            urllib.request.urlretrieve(url, msg_json["img_fn"])
            print("pull img: ",msg_json["img_fn"])
        print("rendered img count:",self.complete_count)
        self.complete_count += 1
        if self.complete_count>=self.max_num:
           self.make_vedio()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        self.ch.basic_consume(self.queue,  #
                              self.on_message)

        self.ch.start_consuming()
if __name__ == '__main__':
    conf_fn = None
    if len(sys.argv) == 1:
        raise Exception('lost config file! ')
    else:
        conf_fn = sys.argv[1]
    base_config = None
    with open(conf_fn, 'r', encoding='utf8')as fp:
        base_config = json.load(fp)
    mq_conf=base_config["mq"]
    io_conf=base_config["io"]


    sim_iter_num = base_config["sim_iter_num"]
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=mq_conf["host_ip"]))
    channel = connection.channel()
    ren_succ_counter_que = "queue_for_render_count"
    channel.queue_declare(queue=ren_succ_counter_que)

    lis =ImgDoneListener(channel,ren_succ_counter_que,sim_iter_num,base_config)
    # lis.start()
    lis.make_vedio()
