import sys
import os
import gym
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import random
import time
import pickle
import numpy
sys.path.insert(0, '/home/carol/libLogHelper/build')
import log_helper as lh
from logger import Logger
from pycoral.utils.edgetpu import make_interpreter

Logger.setLevel(Logger.Level.TIMING)

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> <seed> <iterations> <goldfile> <generate(0|1)>")
        exit(0)
        #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
        seed=int(sys.argv[3])
        iterations=int(sys.argv[4])
        gold_file = sys.argv[5]
        generate = int(sys.argv[6])
    model_save_file = model_prefix + ".tflite"
    env = gym.make(env_name)
    random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    interpreter = make_interpreter(model_save_file)
    interpreter.allocate_tensors()
    lh.start_log_file(env_name, f"repetition:{iterations}")
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if generate == 1:
        gold = open(gold_file,'wb') 
       
    else:
        gold = open(gold_file,'rb')
        golden = pickle.load(gold)
    i = 0
    while i < iterations:
        Logger.info(f"Iteration {i}")
        lh.start_iteration()
        #nn_exec_time=0
        #t0=time.time()
        for j in range(100000):
            input_data = tf.cast(obs.reshape(1, -1),tf.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            #t1=time.time()
            interpreter.invoke()
            #t2=time.time()
            #nn_exec_time+=(t2-t1)
            output_data = interpreter.get_tensor(output_details[0]['index'])
            obs, reward, done, info = env.step(output_data)
            if done:
                lh.end_iteration()
                if i == 3:
                    reward += 1
                random.seed(seed)
                env.seed(seed)
                obs=env.reset()
                #t3=time.time()
                #print(nn_exec_time/(t3-t0))
                i+=1
                if generate == 1:
                    pickle.dump([info,reward],gold)
                    exit(0)
                else: 
                    if not (all([x==y for x,y in zip(golden[0],info)]) and golden[1] == reward):
                        error_detail = f"info: {info} expected info: {golden[0]} reward: {reward} expected reward: {golden[1]}"
                        lh.log_error_detail(error_detail)
                        Logger.error(error_detail)
                        lh.log_error_count(1)
                break

