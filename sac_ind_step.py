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

def create_gold(input_file,gold_file,interpreter,env_name,seed):
    input_data_array=[]
    gold_data=[]
    env = gym.make(env_name)
    random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    for j in range(100000):
        input_data = tf.cast(obs.reshape(1, -1),tf.float32)
        input_data_array.append(input_data)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        #t1=time.time()
        interpreter.invoke()
        #t2=time.time()
        #nn_exec_time+=(t2-t1)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        gold_data.append(output_data)
        obs, reward, done, info = env.step(output_data)
        if done:
            break;
    pickle.dump(gold_data,gold_file)
    pickle.dump(input_data_array,input_file)

if __name__ == '__main__':
    if len(sys.argv) < 8:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> <seed> <iterations> <input_file> <goldfile> <generate(0|1)>")
        exit(0)
        #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
        seed=int(sys.argv[3])
        iterations=int(sys.argv[4])
        input_file=sys.argv[5]
        gold_file = sys.argv[6]
        generate = int(sys.argv[7])
    model_save_file = model_prefix + ".tflite"
    interpreter = make_interpreter(model_save_file)
    interpreter.allocate_tensors()
    lh.start_log_file(env_name, f"repetition:{iterations}")
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if generate == 1:
        gold = open(gold_file,'wb') 
        input_data_file = open(input_file, 'wb')
        create_gold(input_data_file,gold,interpreter,env_name,seed)
        exit(0)
    else:
        gold = open(gold_file,'rb')
        golden = pickle.load(gold)
        input_data_file = open(input_file,'rb')
        input_data=pickle.load(input_data_file)
    i = 0
    while i < iterations:
        Logger.info(f"Iteration {i}")
        error_detail=[]
        lh.start_iteration()
        #nn_exec_time=0
        #t0=time.time()
        for j,(inpt,gld) in enumerate(zip(input_data,golden)):
            #print(inpt)
            interpreter.set_tensor(input_details[0]['index'], inpt)
            #t1=time.time()
            interpreter.invoke()
            #t2=time.time()
            #nn_exec_time+=(t2-t1)
            output_data = interpreter.get_tensor(output_details[0]['index'])
            #if i == 4 and (j == 50 or j == 55):
            #    output_data[0][0]+=1
            #print(output_data)
            if not (all([x==y for x,y in zip(gld[0],output_data[0])])):
                error_detail_init=f"State {j}: "
                error_detail.append(error_detail_init+f" got: {output_data[0]} expected: {gld[0]}")
                Logger.error(error_detail_init+f" got: {output_data[0]} expected: {gld[0]}")
            
        lh.end_iteration()
        #t3=time.time()
        #print(nn_exec_time/(t3-t0))
        i+=1
        if len(error_detail) !=0:
            for k in error_detail:
                lh.log_error_detail(k)
            lh.log_error_count(len(error_detail))
               


