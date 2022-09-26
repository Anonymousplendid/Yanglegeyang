import os
import pickle
from agent import YangLeGeYangeAgent
from env import YangLeGeYangEnv
import numpy as np
import zmq
from pyarrow import serialize

Block_Number = 16
ckpt_path = "D:/yanglegeyang/"


def process(state):
    Bright_Block = np.zeros([1, 20, Block_Number + 2])
    Bright_Block_legal = np.zeros([1, 20])
    l = len(state['Bright_Block'])
    for i in range(l):
        Bright_Block[0][i] = state['Bright_Block'][i]
        Bright_Block_legal[i] = 1
    
    Dark_Block = np.zeros([1, 20, Block_Number + 2])
    Dark_Block_legal = np.zeros([1, 20])
    l = len(state['Dark_Block'])
    for i in range(l):
        Dark_Block[0][i] = state['Dark_Block'][i]
        Dark_Block_legal[i] = 1

    Queue_Block = np.zeros([1, 7, Block_Number])
    Queue_Block_legal = np.zeros([1, 20])
    l = len(state['Queue_Block'])
    for i in range(l):
        Queue_Block[0][i] = state['Queue_Block'][i]
        Queue_Block_legal[i] = 1

    feature = {}
    feature['Bright_Block'] = Bright_Block
    feature['Bright_Block_legal'] = Bright_Block_legal
    feature['Dark_Block'] = Dark_Block
    feature['Dark_Block_legal'] = Dark_Block_legal
    feature['Queue_Block'] = Queue_Block
    feature['Queue_Block_legal'] = Queue_Block_legal
    
    return feature


def find_new_weights(current_model_id, ckpt_path):
    try:
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = pickle.load(f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id



def prepare_training_data(transitions):
    pass


env = YangLeGeYangEnv()
agent = YangLeGeYangeAgent()
state, done = env.reset() # state['Bright_Block']; state['Dark_Block']; state['Queue_Block']
Queue_Block_len = len(state['Queue_Block'])
feature = process(state)

transitions = []

context = zmq.Context()
context.linger = 0  # For removing linger behavior
socket = context.socket(zmq.REQ)
socket.connect(f'tcp://localhost:5000')

while True:
    action = agent.sample(state, feature)
    state, done = env.step(action)
    next_feature = process(feature)
    if len(state['Queue_Block']) < Queue_Block_len:
        reward = 1
    else:
        reward = 0
    Queue_Block_len = len(state['Queue_Block'])
    transitions.append((feature, action, reward, next_feature, done))
    
    if done:
        data = prepare_training_data(transitions)
        transitions.clear()
        socket.send(serialize(data).to_buffer())
        socket.recv()

        new_weights, model_id = find_new_weights(model_id, ckpt_path)
        if new_weights is not None:
            agent.set_weights(new_weights)


