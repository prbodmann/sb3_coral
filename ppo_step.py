import mujoco_py
import gym
import sys
import os
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


if len(sys.argv) < 6:
    print("Usage: " + str(sys.argv[0]) + "<model_prefix> <seed> <iterations> <goldfile> <generate(0|1)>")
    exit(0)
    #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
else:

    model_prefix = env_name = sys.argv[1]
    seed=int(sys.argv[2])
    iterations=int(sys.argv[3])
    gold_file = sys.argv[4]
    generate = int(sys.argv[5])
model_save_file = model_prefix + "_ppo_quant_edgetpu.tflite"
env = gym.make(env_name)
random.seed(seed)
env.seed(seed)
obs = env.reset()
interpreter = make_interpreter(model_save_file)
interpreter.allocate_tensors()
lh.set_max_errors_iter(1001)
lh.start_log_file(env_name, f"repetition:{iterations}")
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
if generate == 1:
    gold = open(gold_file,'wb') 
    golden=[]
else:
    gold = open(gold_file,'rb')
    golden = pickle.load(gold)

i = 0
while i < iterations:
    Logger.info(f"Iteration {i}")
    error_detail=[]
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
        #if i == 4 and (j == 50 or j == 55):
        #    reward+=1
        if generate == 1:
            print(j)
            print(info)
            golden.append([info,reward])
        else: 
            if not (all([x==y for x,y in zip(golden[j][0],info)]) and golden[j][1] == reward):
                if done:
                    error_detail_init=f"Final State {j}: "
                else:
                    error_detail_init=f"Intermediate State {j}: "
                error_detail.append(error_detail_init+f"got info: {info} expected info: {golden[j][0]} and got reward: {reward} expected reward: {golden[j][1]}. Got output: {output_data}")
                #lh.log_error_detail(error_detail)
                Logger.error(error_detail_init+f"got info: {info} expected info: {golden[j][0]} and got reward: {reward} expected reward: {golden[j][1]}. Got output: {output_data}")
        if done:            
            break
    lh.end_iteration()
    if generate:
        Logger.info("Golden created successfully")
        pickle.dump(golden,gold)
        exit(0)
    random.seed(seed)
    env.seed(seed)
    obs=env.reset()
    #t3=time.time()
    #print(nn_exec_time/(t3-t0))
    i+=1
    if len(error_detail) !=0:
        for k in error_detail:
            lh.log_error_detail(k)
        lh.log_error_count(len(error_detail))


