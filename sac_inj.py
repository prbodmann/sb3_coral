import mujoco_py
import gym
import sys
import os
#import gym
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import random
import time
import pickle
import numpy
#sys.path.insert(0, '/home/carol/libLogHelper/build')
#import log_helper as lh
#from logger import Logger
from pycoral.utils.edgetpu import make_interpreter
#Logger.setLevel(Logger.Level.TIMING)
#sys.path.insert(0, '/home/carol/mujoco-py')
#import mujoco_py
from PIL import Image
import PIL

if len(sys.argv) < 4:
    print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> <seed>")
    exit(0)
    #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
else:
    env_name = sys.argv[1]
    model_prefix = sys.argv[2]
    seed=int(sys.argv[3])
model_save_file = model_prefix + ".tflite"
env = gym.make(env_name)
random.seed(seed)
env.seed(seed)
obs = env.reset()
interpreter = make_interpreter(model_save_file)
interpreter.allocate_tensors()
#lh.start_log_file(env_name, f"repetition:{iterations}")
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for j in range(100000):
    input_data = tf.cast(obs.reshape(1, -1),tf.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    #t1=time.time()
    interpreter.invoke()
    #t2=time.time()
    #nn_exec_time+=(t2-t1)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    obs, reward, done, info = env.step(output_data)
    im1=Image.fromarray(env.render())
    im1.save(model_prefix+"_"+str(j)+".jpg")
    if done:
        break
