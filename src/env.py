import win32gui
import sys
import numpy as np
import win32con
import time
import datetime
from PIL import Image, ImageGrab
from utils import *
import logging
label_num = 16
waiting_time = 2.2
stopping_time_for_next_step = 0.5
class Env():
    def __init__(self) -> None:
        pass
    def reset(self):
        pass

    def step(self):
        pass

class YangLeGeYangEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.hwnd = win32gui.FindWindow(None, '羊了个羊')
        self.pos = win32gui.GetWindowRect(self.hwnd)
        if not self.hwnd:
            print("羊了个羊未打开")
        t = datetime.datetime.now().strftime('%b%d.%H-%M-%S')
        logger = logging.getLogger(t + "env")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler("log/" + t + "env" + ".txt")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        self.logger = logger

    def get_first_level(self):
        for i in range(2):
            posmap, img_data = get_obs(self.hwnd, self.pos)
            clickall(posmap)
        time.sleep(1)

    def obs_process(self, posmap):
        bright_obs = []
        for i in range(label_num):
            for j in range(len(posmap[i])):
                bright_obs.append((posmap[i][j][0], posmap[i][j][1], i))
        return bright_obs

    def reset(self):
        if not self.hwnd:
            self.hwnd = win32gui.FindWindow(None, '羊了个羊')
        if not self.hwnd:
            print("羊了个羊未打开")
            return None
        img, t = get_img(self.hwnd, self.pos)
        img = img/255
        delta = loss_mse(img, restartdata)
        if delta < threshold_restart:
            clickrestart(self.hwnd, self.pos)
        delta = loss_mse(img, endingdata)
        print("restart delta, ", delta)
        if delta < threshold_ending:
            print("restart!")
            time.sleep(1)
            clickending(self.hwnd, self.pos)
            time.sleep(2)
            img, t = get_img(self.hwnd, self.pos)
            img = img/255
            delta = loss_mse(img, problemdata)
            if delta < threshold_problem:
                print("click problem!")
                clickproblem()
            else:
                delta = loss_mse(img, baddata)
                if delta < threshold_bad:
                    self.logger.info("bad ok!")
                    clickbad(self.hwnd, self.pos)
            print("waiting for restart")
            time.sleep(waiting_time)
            clickrestart(self.hwnd, self.pos)
        delta = loss_mse(img, restartdata)
        if delta < threshold_restart:
            clickrestart(self.hwnd, self.pos)
        delta = loss_mse(img, menudata)
        print("menudelta: ", delta)
        if delta < threshold_menu:
            print("menu!")
            clickstart(self.hwnd, self.pos)
        time.sleep(4)
        self.get_first_level()   
        time.sleep(1)
        bright_obs, queue_obs, dark_obs, img_data, t = get_real_obs(self.hwnd, self.pos)
        # print(bright_obs)
        obs = dict()
        obs["Bright_Block"] = bright_obs
        obs["Dark_Block"] = dark_obs
        obs["Queue_Block"] = queue_obs
        self.logger.info(t)
        self.logger.info("Bright_Block: {}".format(str(bright_obs)))
        self.logger.info("Dark_Block: {}".format(str(dark_obs)))
        self.logger.info("Queue_Block: {}".format(str(queue_obs)))
        self.logger.info("\n\n")
        return obs, 0, img_data
    

    def step(self, pos):
        win32gui.SendMessage(self.hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(self.hwnd)
        MouseClick(pos)
        time.sleep(0.5)
        print("before obs")
        bright_obs, queue_obs, dark_obs, img_data, t = get_real_obs(self.hwnd, self.pos)
        print("after obs")
        done = 0
        delta = loss_mse(img_data, endingdata)
        if delta < threshold_ending:
            done = 1
        delta = loss_mse(img_data, baddata)
        if delta < threshold_bad:
            done = 1
        delta = loss_mse(img_data, restartdata)
        if delta < threshold_restart:
            done = 1
        delta = loss_mse(img_data, donedata)
        if delta < threshold_done:
            done = 1
        if done:
            time.sleep(1)
        obs = dict()
        obs["Bright_Block"] = bright_obs
        obs["Dark_Block"] = dark_obs
        obs["Queue_Block"] = queue_obs
        self.logger.info(t)
        self.logger.info("Bright_Block: {}".format(bright_obs))
        self.logger.info("Dark_Block: {}".format(dark_obs))
        self.logger.info("Queue_Block: {}".format(queue_obs))
        self.logger.info("\n\n")
        return obs, done, img_data

if __name__ == "__main__":
    env = YangLeGeYangEnv()
    obs, done, info = env.reset()
    while True:
        idx = np.random.randint(len(obs["Bright_Block"]))
        obs, done, img_data = env.step((obs["Bright_Block"][idx][0], obs["Bright_Block"][idx][1]))
        if done:
            obs, done, info = env.reset()