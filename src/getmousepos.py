import time
import pyautogui as pag
 
try:
    while True:
    	#获取屏幕分辨率
        screenWidth, screenHeight = pag.size()  
        #获取鼠标位置
        x, y = pag.position()  
        #打印分辨率和鼠标位置
        print("Screen size: (%s %s),  Position : (%s, %s)\n" % (screenWidth, screenHeight, x, y))  
        #间隔一秒显示位置
        time.sleep(1)  
except KeyboardInterrupt:
    print('end')
