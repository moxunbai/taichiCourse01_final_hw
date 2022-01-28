# 太极第一季大作业


## 背景简介
大作业主要做的是模拟仿真‘纸飞机’在‘空气中’飞行(后补充流固耦合仿真)，并渲染成视频，其中仿真部分是用的[taichi_elements](https://github.com/taichi-dev/taichi_elements)的代码；渲染部分是再之前作业的基础上优化调整后的。并实现了一个简单的基于RabbitMQ消息中间件的分布式方案来均衡利用多台主机（本地一台台式机一台笔记本）的GPU算力来更快的仿真、渲染、合成视频；项目详情请查看[太极图形论坛](https://forum.taichi.graphics/t/topic/2246)提交信息。
## 渲染效果 
[纸飞机](https://www.bilibili.com/video/BV1XT4y127Yu/ )
[液体仿真](https://www.bilibili.com/video/BV1Uu41117QN/ )
## 依赖模块
taichi、numpy、pillow、pika、opencv-python  
## 运行环境
系统windows10 、python 3.6.1、taichi 0.8.8 ；如果想尝试分布式生成‘纸飞机’的需要安装RabbitMQ

## 整体结构（Optional）
```
-|config 
   base.json 分布式任务执行配置文件
   -re_conf_cornellbox.json 康纳尔盒子场景描述文件
   -re_conf_demo1.json 环境贴图场景描述文件
   -re_conf_pplan.json 纸飞机单个图片渲染的场景描述文件（测试用的，若要使用要改里面模型文件的地址）
   -re_conf_pplan_sim.json 仿真渲染纸飞机的模板场景描述文件
-|data  模型贴图等数据
-|engine taichi_elements的代码
-|loader 加载模型文件工具
-|render 渲染程序相关
-main_task.py 分布式任务主程序

-README.MD
-render_consumer.py 基于mq渲染的消费者，为main_task.py调用
-sim_paper_plan.py 调用taichi_elements的mpm_solver仿真的程序为main_task.py调用
-test_env_light.py 测试环境光渲染的，不用场景描述文件的方式
-test_render_entry.py 使用场景描述文件渲染单个图片的
-utils.py taichi_elements的工具
-video_task.py 合成视频的工具
```

## 渲染demo
* cornellbox渲染
   参考命令： python test_render_entry.py config/re_conf_cornellbox.json
* 环境贴图渲染  
   参考命令：python test_render_entry.py  config/re_conf_demo1.json
 
   
## 生成流体仿真渲染视频步骤
* 整个任务分为仿真、渲染、视频合成三部分；
* 如果是单台主机基本三部分就是顺序执行了，配置也简单只用修改下配置文件config/task_water_flow.json中的 mq地址即可，改好配置分别运行:python task_water_flow.py ;python video_task.py config/task_water_flow.json即可，可以等待前者执行完在执行后者，也可以同时执行
(配置参数中要仿真渲染800帧，很慢，可调小)
* 如果是部署多台（假设2台局域网的主机，一台‘主’，一台‘次’），‘主’三种任务都可以做，‘次’主要做渲染
1. 先在两台的程序目录执行python -m http.server 启动http服务，用来做中间文件传输同步；
2. 在‘主’执行机器（A）上配置，在上面单台配置的基础上，修改http_server_host项为本机局域网ip
3. 在‘次’机器上除了上一步外，将task_scope项改为"render",将io.input_source项改为"remote"
4. 然后两台机器分别执行main_task.py，再随便一台执行video_task.py即可
