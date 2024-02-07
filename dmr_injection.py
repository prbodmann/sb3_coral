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
from pycoral.utils.edgetpu import make_interpreter
descision_dict = {
"Walker2d-v3": [1,	1,	1,	1,	1,	1],
"Hopper-v3":[-1,1,1],
"Humanoid-v3":[1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	-1,	-1],
"HalfCheetah-v3":[]
}

 limit_dict = {
    "Walker2d-v3":[-10,10],
    "Hopper-v3":[-90,90],
    "Humanoid-v3":[-50,50],
    "HalfCheetah-v3":[-10,10]
}



if len(sys.argv) < 7:
    print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> <seed> <number of injections>")
    exit(0)
    #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
else:
    env_name = sys.argv[1]
    model_prefix = sys.argv[2]
    seed=int(sys.argv[3])
    num_injections=int(sys.argv[4])

model_save_file = model_prefix + ".tflite"
env_dmr = gym.make(env_name)
random.seed(seed)
env_dmr.seed(seed)
obs = env_dmr.reset()

env_not_protected = gym.make(env_name)
random.seed(seed)
env_not_protected.seed(seed)
obs = env_not_protected.reset()

interpreter_dmr1 = make_interpreter(model_save_file)
interpreter_dmr2 = make_interpreter(model_save_file)
interpreter_not_protected = make_interpreter(model_save_file)

interpreter_dmr1.allocate_tensors()
interpreter_dmr2.allocate_tensors()
interpreter_not_protected.allocate_tensors()
lh.set_max_errors_iter(1001)
lh.start_log_file(env_name, f"repetition:{iterations}")
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

first_errouneous_step = random.randint(0, 1000)

i = 0
step_counter_dmr = 0
step_counter_np = 0

while i < num_injections:

    for j in range(1001):
        input_data = tf.cast(obs.reshape(1, -1),tf.float32)
        
        if j>=first_errouneous_step:

            for i in random.randint(0, len(input_data)):
                array_inex=random.randint(0, len(input_data))
                wrong_array = input_data
                wrong_array[array_inex] += np.random.uniform(-100,0,100)

            input_data_not_protected = wrong_array
            if random.randint(0, 1) == 0:
                interpreter_dmr1 = wrong_array
                interpreter_dmr2 = input_data
            else:
                interpreter_dmr1 = input_data
                interpreter_dmr2 = wrong_array
    
        if not done_dmr:
            interpreter_dmr1.set_tensor(input_details[0]['index'], input_data_dmr1)
            interpreter_dmr2.set_tensor(input_details[0]['index'], input_data_dmr2)
            interpreter_dmr1.invoke()
            interpreter_dmr2.invoke()
            output_data_dmr1 = interpreter_dmr1.get_tensor(output_details[0]['index'])
            output_data_dmr2 = interpreter_dmr2.get_tensor(output_details[0]['index'])
            #seleciona core
            obs_dmr, reward_dmr, done_dmr, inf_dmr = env_dmr.step(output_data_dmr)
            step_counter_dmr += 1
        if not done_np:
            interpreter_not_protected.set_tensor(input_details[0]['index'], input_data_not_protected)
            interpreter_not_protected.invoke()
            output_data_not_protected = interpreter_not_protected.get_tensor(output_details[0]['index'])
            obs_np, reward_np, done_np, info_np = env_not_protected.step(output_data_not_protected)
            step_counter_np += 1


        if done_np and done_dmr:            
            print(f"dmr: info {inf_dmr} num_steps: {step_counter_dmr} ----- not_proteted: info: {info_np} num_steps: {step_counter_np}")
            break

   
