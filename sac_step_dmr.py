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
from threading import Thread

sys.path.insert(0, '/home/carol/libLogHelper/build')
import log_helper as lh
from logger import Logger
from pycoral.utils.edgetpu import make_interpreter
Logger.setLevel(Logger.Level.TIMING)

def thread_func(interpreter, input_data, input_details):
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

if len(sys.argv) < 5:
    print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> <seed> <iterations>")
    exit(0)
    #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
else:
    env_name = sys.argv[1]
    model_prefix = sys.argv[2]
    seed=int(sys.argv[3])
    iterations=int(sys.argv[4])

model_save_file = model_prefix + ".tflite"
env = gym.make(env_name)
random.seed(seed)
env.seed(seed)
obs = env.reset()

interpreter1=make_interpreter(model_save_file,device=':0')
interpreter1.allocate_tensors()
interpreter2=make_interpreter(model_save_file,device=':1')
interpreter2.allocate_tensors()
lh.start_log_file(env_name, f"repetition:{iterations}")
# Get input and output tensors.
input_details = interpreter1.get_input_details()
output_details = interpreter1.get_output_details()

i = 0
while i < iterations:
    Logger.info(f"Iteration {i}")
    error_detail=[]
    lh.start_iteration()
    #nn_exec_time=0
    #t0=time.time()
    for j in range(100000):
        input_data = tf.cast(obs.reshape(1, -1),tf.float32)
        t1=Thread(target=thread_func, args=(interpreter1,input_data,input_details))
        t2=Thread(target=thread_func, args=(interpreter2,input_data,input_details))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        #t2=time.time()
        #nn_exec_time+=(t2-t1)
        output_data = interpreter1.get_tensor(output_details[0]['index'])
        golden = interpreter2.get_tensor(output_details[0]['index'])
        #how to choose which one is correct??
        obs, reward, done, info = env.step(output_data)
        
        if not (output_data==golden).all():
            if done:
              error_detail_init=f"Final State {j}: "
            else:
              error_detail_init=f"Intermediate State {j}: "
            error_detail.append(error_detail_init+f"got from tpu 0: {output_data} and got from tpu 1: {golden}.")
            #lh.log_error_detail(error_detail)
            Logger.error(error_detail_init+f"got from tpu 0: {output_data} and got from tpu 1: {golden}")
        #save env output?
        #if i == 4 and (j == 50 or j == 55):
        #    reward+=1
        if done:            
            break
    lh.end_iteration()
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
