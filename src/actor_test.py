from env import YangLeGeYangEnv
import random
import numpy as np
import logging
from utils import *

step_watcher = AverageMeter()
score = AverageMeter()

env = YangLeGeYangEnv()
logger = env.logger
state, done, info = env.reset() # state['Bright_Block']; state['Dark_Block']; state['Queue_Block']
Queue_Block_len = len(state['Queue_Block'])
step = 0
print("reset done!")
q = []
reward_sum = 0
while True:
    print(q)
    print(state['Bright_Block'])
    print(state['Dark_Block'])
    print(state['Queue_Block'])
    if len(q) != 0:
        action = (q[0][0], q[0][1])
        q.pop(0)
    else:
        d = []
        dd = []
        for i in range(16):
            dd.append(0)
        for i in range(len(state['Queue_Block'])):
            dd[state['Queue_Block'][i][2]] += 1

        for i in range(16):
            d.append([])
        for i in range(len(state['Bright_Block'])):
            d[state['Bright_Block'][i][2]].append((state['Bright_Block'][i][0], state['Bright_Block'][i][1]))
            if len(d[state['Bright_Block'][i][2]]) + dd[state['Bright_Block'][i][2]] >= 3:
                for j in range(3-dd[state['Bright_Block'][i][2]]):
                    q.append(d[state['Bright_Block'][i][2]][j])
                break
        logger.info(q)
        if len(q)==0:
            idx = np.random.randint(len(state["Bright_Block"]))
            action = (state["Bright_Block"][idx][0], state["Bright_Block"][idx][1])
        else:
            action = (q[0][0], q[0][1])
            q.pop(0)
    logger.info(q)
    state, done, info = env.step(action)
    if len(state['Queue_Block']) < Queue_Block_len:
        reward = 1
    else:
        reward = 0
    Queue_Block_len = len(state['Queue_Block'])
    reward_sum += reward
    step += 1
    if done:
        score.update(reward_sum)
        reward_sum = 0
        step_watcher.update(step)
        logger.info("avg: {} for time {}".format(step_watcher.avg, step_watcher.cnt))
        print("avg: {} for time {}".format(step_watcher.avg, step_watcher.cnt))
        logger.info("avg: {} for time {}".format(score.avg, score.cnt))
        print("avg: {} for time {}".format(score.avg, score.cnt))
        state, done, info = env.reset()

 