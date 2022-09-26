import multiprocessing
import pickle
from itertools import count
from multiprocessing import Process
from agent import YangLeGeYangeAgent
import tensorflow as tf
import zmq
from pyarrow import deserialize
from mem_pool import MemPoolManager, MultiprocessingMemPool


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.set_session(tf.Session(config=config))
pool_size = 1000
batch_size = 128

def main():
    # Parse input parameters

    context = zmq.Context()
    weights_socket = context.socket(zmq.PUB)
    weights_socket.bind(f'tcp://*:5001')

    agent = YangLeGeYangeAgent()

    receiving_condition = multiprocessing.Condition()
    num_receptions = multiprocessing.Value('i', 0)

    # Start memory pool in another process
    manager = MemPoolManager()
    manager.start()
    mem_pool = manager.MemPool(capacity=pool_size)
    Process(target=recv_data,
            args=(5000, mem_pool, receiving_condition, num_receptions, False)).start()

    # Print throughput statistics
    Process(target=MultiprocessingMemPool.record_throughput, args=(mem_pool, 10)).start()

    for step in count(1):
        if len(mem_pool) >= batch_size:
            with receiving_condition:
                while num_receptions.value < 1:
                    receiving_condition.wait()
                data = mem_pool.sample(size=batch_size)
                num_receptions.value -= 1
            # Training
            agent.learn(data)

            # Sync weights to actor
            weights_socket.send(pickle.dumps(agent.get_weights()))


def recv_data(data_port, mem_pool, receiving_condition, num_receptions, keep_training):
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{data_port}')

    while True:
        # noinspection PyTypeChecker
        data: dict = deserialize(data_socket.recv())
        data_socket.send(b'200')

        if keep_training:
            mem_pool.push(data)
        else:
            with receiving_condition:
                mem_pool.push(data)
                num_receptions.value += 1
                receiving_condition.notify()


if __name__ == '__main__':
    main()
