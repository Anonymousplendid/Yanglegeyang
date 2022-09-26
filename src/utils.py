
import queue
from PyQt5.QtWidgets import QApplication, QLabel
import win32gui
import sys
import cv2
import numpy as np
import win32con
import win32api
import win32print
from ctypes import windll
import time
import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
import copy
from tqdm import tqdm
from skimage.metrics import structural_similarity

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def is_empty(self):
        return self.cnt == 0
    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum / self.cnt

sX = win32api.GetSystemMetrics(0)
hDC = win32gui.GetDC(0)
w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
scale = w / sX

label_num = 16

start = 0.9
ending = 0.9
restart = 0.8
bad = 0.71
upcrop = 150 # for recognizing light
downcrop = 560
label_size = 60
step = 2
# TODO: a menu model and a corresponding threshold
menuimg = Image.open("label/menu.png")
wid, hei = menuimg.size
menudata = menuimg.getdata()
menudata = np.array(menudata) / 255
menudata = menudata.reshape((hei, wid, 3))
threshold_menu = 0.12

# TODO: a ending model and restart model and corresponding threshold
endingimg = Image.open("label/ending.png")
wid, hei = endingimg.size
endingdata = endingimg.getdata()
endingdata = np.array(endingdata) / 255
endingdata = endingdata.reshape((hei, wid, 3))
threshold_ending = 0.12

# TODO: a done signal image
doneimg = Image.open("label/done.png")
wid, hei = doneimg.size
donedata = doneimg.getdata()
donedata = np.array(donedata) / 255
donedata = donedata.reshape((hei, wid, 3))
threshold_done = 0.12

# TODO: a bad image model
badimg = Image.open("label/bad.png")
wid, hei = badimg.size
baddata = badimg.getdata()
baddata = np.array(baddata) / 255
baddata = baddata.reshape((hei, wid, 3))
threshold_bad = 0.12

problemimg = Image.open("label/problem.png")
wid, hei = problemimg.size
problemdata = problemimg.getdata()
problemdata = np.array(problemdata) / 255
problemdata = problemdata.reshape((hei, wid, 3))
threshold_problem = 0.12

restartimg = Image.open("label/restart.png")
wid, hei = restartimg.size
restartdata = restartimg.getdata()
restartdata = np.array(restartdata) / 255
restartdata = restartdata.reshape((hei, wid, 3))
threshold_restart = 0.12

def getcards(im):
#返回一个numpy数组，每一行包含：label,x,y,w,h,center_x,center_y
#用法见52行
    cards = []
    flag = np.mean(np.abs(im - np.array([245,255,205])), 2) < 15#获取卡牌背景
    flag = np.array(flag, dtype='uint8')

    imgs = np.zeros((16,45,45,3), dtype='uint8')#读取标注模板
    for i in range(16):
        imgs[i] = np.array(Image.open('img/{:}.png'.format(i)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(flag, connectivity=8)

    for i in range(num_labels):
        if stats[i,4] > 500 and i != 0:#判定为卡牌
            x,y,w,h = stats[i,:4]
            center_x,center_y = centroids[i]

            img = im[y:y+h,x:x+w]#将判定为卡牌的区域单独取出来作为img
            img = np.array(Image.fromarray(img).resize((45,45)))

            ssmi = np.zeros(16)
            for j in range(16):#将img和标签逐个比较
                ssmi[j] = structural_similarity(img, imgs[j], data_range=255, multichannel=True)
            label = np.argmax(ssmi)

            cards.append([label,x,y,w,h,int(center_x),int(center_y)])#存放如cards

    cards = np.array(cards)
    return cards

def getcards_dark(im):
#返回一个numpy数组，每一行包含：label,x,y,w,h,center_x,center_y
#用法见52行
    cards = []
    flag = np.mean(np.abs(im - np.array([147,153,123])), 2) < 10#获取卡牌背景
    flag = np.array(flag, dtype='uint8')

    imgs = np.zeros((16,45,45,3), dtype='uint8')#读取标注模板
    for i in range(16):
        imgs[i] = np.array(Image.open('img/{:}.png'.format(i)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(flag, connectivity=8)

    for i in range(num_labels):
        if stats[i,4] > 250 and i != 0:#判定为卡牌
            x,y,w,h = stats[i,:4]
            center_x,center_y = centroids[i]

            img = im[y:y+h,x:x+w]#将判定为卡牌的区域单独取出来作为img
            img = np.array(Image.fromarray(img).resize((45,45)))

            ssmi = np.zeros((16,8))
            for j in range(16):#将img和标签逐个比较
                if h/w > 1.5:
                    temp = np.array(Image.fromarray(imgs[j][:,:22]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,0] = structural_similarity(img  , temp, data_range=255, multichannel=True)
                    
                    temp = np.array(Image.fromarray(imgs[j][:, 22:]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,1] = structural_similarity(img  , temp, data_range=255, multichannel=True)

                elif h/w < 0.75:
                    temp = np.array(Image.fromarray(imgs[j][:22,:]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,2] = structural_similarity(img  , temp, data_range=255, multichannel=True)

                    temp = np.array(Image.fromarray(imgs[j][22:,:]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,3] = structural_similarity(img  , temp, data_range=255, multichannel=True)

                else:
                    temp = np.array(Image.fromarray(imgs[j][:22,:22]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,4] = structural_similarity(img  , temp, data_range=255, multichannel=True)

                    temp = np.array(Image.fromarray(imgs[j][:22,22:]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,5] = structural_similarity(img  , temp, data_range=255, multichannel=True)

                    temp = np.array(Image.fromarray(imgs[j][22:,:22]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,6] = structural_similarity(img  , temp, data_range=255, multichannel=True)

                    temp = np.array(Image.fromarray(imgs[j][22:,22:]).resize((45,45)))#imgs[j]各个方向的一半图片
                    ssmi[j,7] = structural_similarity(img  , temp, data_range=255, multichannel=True)
            if np.max(ssmi) > 0.45:
                label = np.argmax(ssmi) // 8
                cards.append([label,x,y,w,h,int(center_x),int(center_y)])#存放如cards

    cards = np.array(cards)
    return cards

def get_index(im, i):
#返回一个shape为（n,2）的列表[[x,y],[x,y],,,,,[x,y]]，n为检测到的个数
# 其中[x,y]分别是检测到的卡片左上角的坐标
    color = np.array([
        [255,200,50],
        [8,113,231],
        [232,220,194],
        [255,72,72],
        [170,223,67],
        [255,165,233],
        [255,240,113],#6
        [73,154,217],
        [187,104,43],
        [255,165,49],
        [148,205,38],
        [19,86,255],
        [255,240,0],
        [163,98,35],
        [255,242,41],
        [255,108,108]
    ])

    seed_index = np.array([
        [17, 22],
        [ 6, 24],
        [ 20, 36],
        [ 20, 29],
        [ 6, 19],
        [ 6, 24],
        [ 5,20],
        [12, 19],
        [ 4, 29],
        [11, 21],
        [ 9, 15],
        [ 3, 25],
        [11, 15],
        [14,  38],#13
        [ 7, 20],
        [26, 21],
    ])

    flag = np.mean(np.abs(im - color[i]), 2) <= 1
    index = []
    while np.max(flag) != 0:
        x = np.argmax(flag)//450 - seed_index[i,0]
        y = np.argmax(flag)%450 - seed_index[i,1]
        flag[x-5:x+45, y-5:y+45] = False
        index.append([x,y])
    return index


def loss_mse(data1, data2):
    assert data1.shape == data2.shape, (data1.shape, data2.shape)
    return np.square(np.subtract(data1, data2)).mean()

def MouseClick(pos):
    x, y = pos
    x = int(x)
    y = int(y)
    windll.user32.SetCursorPos(x, y)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def get_img(hwnd, pos):
    if hwnd:
        win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hwnd)
        x1, y1, x2, y2 = pos
        x1 = x1 * scale
        y1 = y1 * scale
        x2 = x2* scale
        y2 = y2 * scale
        time.sleep(0.1)
        img = ImageGrab.grab((x1, y1, x2, y2))
        # wid, ori_hei = img.size
        # upcrop = int(up * ori_hei)
        # img = img.crop((0, upcrop, wid, ori_hei))
        t = datetime.datetime.now().strftime('%b%d.%H-%M-%S')
        # img.save("picture/{}.png".format(t))
        # print(img.size)
        #input()
        img = img.resize((450,844))
        wid, hei = img.size
        img.save("picture/{}.png".format(t))
        data = np.array(img)
        wid, hei = img.size
        data = data.reshape(hei, wid, 3)
        delta = loss_mse(data/255, endingdata)
        # print(delta)
        return data, t
        # print(delta)
    else:
        print("羊了个羊未打开！")

def clickstart(hwnd, pos):
    if hwnd:
        win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hwnd)
        x1, y1, x2, y2 = pos
        posx = (x2 + x1) / 2
        posy = start * (y2 - y1) + y1
        MouseClick((posx, posy))

def clickending(hwnd, pos):
    if hwnd:
        win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        MouseClick((955, 704))

def clickrestart(hwnd, pos):
    if hwnd:
        win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hwnd)
        x1, y1, x2, y2 = pos
        posx = (x2 + x1) / 2
        posy = restart * (y2 - y1) + y1
        MouseClick((posx, posy))
    
def clickbad(hwnd, pos):
    if hwnd:
        win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)
        MouseClick((1103, 318))

def clickproblem():
    MouseClick((1217, 306))

def get_obs(hwnd, pos):
    img_data, t = get_img(hwnd, pos)
    ori_img_data = copy.deepcopy(img_data)
    x1, y1, x2, y2 = pos
    # plt.imshow(img_data)
    # plt.show()
    img_data = img_data[upcrop:downcrop]
    # print(img_data.shape)
    # plt.imshow(img_data)
    # plt.show()
    posmap = dict()
    for i in tqdm(range(label_num)):
        print("before getindex_{}".format(i))
        cur_index = get_index(img_data, i)
        print("get_index_{} done".format(i))
        for j in range(len(cur_index)):
            cur_index[j][0] += 150
            temp = cur_index[j][0]
            cur_index[j][0] = cur_index[j][1]
            cur_index[j][1] = temp
            cur_index[j][0] = (cur_index[j][0] + 22.5)/450 * (x2-x1)+x1
            cur_index[j][1] = (cur_index[j][1] + 22.5)/844 * (y2-y1)+y1
        posmap[i] = cur_index
    return posmap, ori_img_data/255
    
def clickall(posmap):
    level1_set = set([9, 10, 12])
    for i in range(label_num):
        if not i in level1_set:
            continue
        cur_pos_list = posmap[i]
        for j in range(len(cur_pos_list)):
            MouseClick(cur_pos_list[j])
            time.sleep(0.25)

def get_real_obs(hwnd, pos):
    img_data, t = get_img(hwnd, pos)
    x1, y1, x2, y2 = pos
    # plt.imshow(img_data)
    # plt.show()
    main_data = img_data[upcrop:downcrop]
    queue_data = img_data[downcrop:]
    bright_cards = getcards(main_data)
    queue_cards = getcards(queue_data)
    dark_cards = getcards_dark(main_data)
    bright_obs = []
    queue_obs = []
    dark_obs = []
    for i in range(len(bright_cards)):
        label,x,y,w,h,center_x,center_y = bright_cards[i]
        center_x = x + w/2
        center_y = y + h/2
        center_x = center_x/450 * (x2-x1)+x1
        center_y = (center_y + upcrop)/844 * (y2-y1)+y1
        bright_obs.append((center_x, center_y, label))
    for i in range(len(queue_cards)):
        label,x,y,w,h,center_x,center_y = queue_cards[i]
        center_x = x + w/2
        center_y = y + h/2
        center_x = center_x/450 * (x2-x1)+x1
        center_y = (center_y + downcrop)/844 * (y2-y1)+y1
        queue_obs.append((center_x, center_y, label))
    for i in range(len(dark_cards)):
        label,x,y,w,h,center_x,center_y = dark_cards[i]
        center_x = x + w/2
        center_y = y + h/2
        center_x = center_x/450 * (x2-x1)+x1
        center_y = (center_y + upcrop)/844 * (y2-y1)+y1
        dark_obs.append((center_x, center_y, label))
    return bright_obs, queue_obs, dark_obs, img_data/255, t

if __name__ == "__main__":
    hwnd = win32gui.FindWindow(None, '羊了个羊')
    pos = win32gui.GetWindowRect(hwnd)
    clickending(hwnd, pos)
