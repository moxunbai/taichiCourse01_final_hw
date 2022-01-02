# 太极第一季大作业-“纸飞机”
大作业主要做的是模拟仿真‘纸飞机’在‘空气中’飞行，并渲染成视频，其中仿真部分是用的[taichi_elements](https://github.com/taichi-dev/taichi_elements)的代码；渲染部分是再之前作业的基础上优化调整后的。并实现了一个简单的基于RabbitMQ消息中间件的分布式方案来均衡利用多台主机（本地一台台式机一台笔记本）的GPU算力来更快的仿真、渲染、合成视频；项目详情请查看[太极图形论坛]( )提交信息。


## 渲染效果 
[纸飞机]( )
 
## 依赖模块
taichi、numpy、pillow、pika、opencv-python  
## 运行环境
系统windows10 、python 3.6.1、taichi 0.8.4 ；如果想尝试分布式生成‘纸飞机’的需要安装RabbitMQ
  
## 渲染demo
* cornellbox渲染
   参考命令： python test_render_entry.py re_conf_cornellbox.json 
* 环境贴图渲染  
   参考命令：python test_render_entry.py  re_conf_demo1.json 
 
   
## 生成‘纸飞机’步骤 
* 整个任务分为仿真、渲染、视频合成三部分；
* 如果是单台主机基本三部分就是顺序执行了，配置也简单只用修改下配置文件config/base.json中的 mq地址即可，改好配置分别运行main_task.py ;video_task.py即可，可以等待前者执行完在执行后者，也可以同时执行

* 如果是部署多台（假设2台局域网的主机，一台‘主’，一台‘次’），‘主’三种任务都可以做，‘次’主要做渲染
1. 先在两台的程序目录执行python -m http.server 启动http服务，用来做中间文件传输同步；
2. 在‘主’执行机器（A）上配置，在上面单台配置的基础上，修改http_server_host项为本机局域网ip
3. 在‘次’机器上除了上一步外，将task_scope项改为"render",将io.input_source项改为"remote"
4. 然后两台机器分别执行main_task.py，再随便一台执行video_task.py即可
