# 非侵入式深度强化学习AI：羊了个羊
还在被羊困扰吗，来看看人工智能在羊了个羊上的表现吧！****记得star给予鼓励呀！
较为详细的介绍可参见https://zhuanlan.zhihu.com/p/568307169。
合作者：赵鉴、薛东昀、姚金辰。
## 安装
clone 本仓库并 安装requirement中所有库
## 玩游戏
1. 在电脑上打开微信小程序“羊了个羊”，若是当天第一次打开，请先玩一次并使页面停留在死亡界面
2. 在根目录下运行actor_test.py程序即可使用固定规则（看到3个相同的立刻消掉，否则随机选择放入队列）和小程序交互。

## 开发
1. src/utils和env中封装了强化学习YangLeGeYangEnv环境类。提供以下接口：
-  reset 初始化小程序，初始化前游戏状态仅支持死亡后的任意界面和开始界面， 输出obs, done, info
-  step 输入一个表示鼠标点击位置的元组，输出obs, done, info
  
2. 以上信息中：
-  obs为状态表示, 格式为一个python dict
-  done确认了游戏是否结束，结束则为1，否则为0
-  info为当前游戏帧的截图，可以查看

1. obs
obs全部通过形态学、cv等方法从游戏截图中识别。
-  Bright_Block
   在最表层可以点击的亮块。格式为一个python list，其中每一个元素为一个元组，格式为(posx, posy, label)。label代表了亮块的内容，具体课件img文件夹。(posx, posy)为位置，即对应亮块的可选鼠标点击动作位置。
-  Queue_Block
   在底部的队列块，格式与Bright_Block相同
-  Dark_Block 
   隐藏的暗块，格式与Bright_Block相同

4. 各个接口的使用范例可见src/env，在根目录直接运行该文件可以看到随机策略的效果。
 
