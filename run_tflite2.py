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
    
    #delegates = None
    #if 'edgetpu' in model_save_file:
    #    print("using tpu")
    delegates = [tflite.load_delegate('libedgetpu.so.1')]
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    env = gym.make(env_name)
    random.seed(seed)
    env.seed(seed)
    #tf.keras.utils.set_random_seed(seed)
    #tf.config.experimental.enable_op_determinism()
    obs = env.reset()
    #interpreter = tflite.Interpreter(model_path=model_save_file, experimental_delegates=delegates,num_threads=1)
    from pycoral.utils.edgetpu import make_interpreter
    interpreter = make_interpreter(model_save_file)
    interpreter.allocate_tensors()
    lh.start_log_file(env_name, f"repetition:{iterations}")
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #input_details=tf.cast(input_details,tf.float32)
    if generate == 1:
        gold = open(gold_file,'wb') 
        golden=[]
    else:
        gold = open(gold_file,'rb')
        golden = pickle.load(gold)
        
    #start=time.time()
    i = 0
    while i < iterations:
        Logger.info(f"Iteration {i}")
        lh.start_iteration()
        nn_exec_time=0
        err_count=0
        t0=time.time()
        for j in range(100000):
            input_data = tf.cast(obs.reshape(1, -1),tf.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t1=time.time()
            interpreter.invoke()
            t2=time.time()
            nn_exec_time+=(t2-t1)
            output_data = interpreter.get_tensor(output_details[0]['index'])
            #t3=time.time()
            obs, reward, done, info = env.step(output_data)
            #if i == 4 and j == 50:
            #    reward+=1
            if generate == 1:
                #pickle.dump([info,reward],gold)
                golden.append([info,reward])
            else: 
                #print([x==y for x,y in zip(golden[0],obs)])
                if not (all([x==y for x,y in zip(golden[j][0],info)]) and golden[j][1] == reward):
                    if done:
                        error_detail="Final State: "
                    else:
                        error_detail="Intermediate State: "
                    error_detail += f"step: {j} info: {info} expected info: {golden[j][0]} reward: {reward} expected reward: {golden[j][1]}"
                    lh.log_error_detail(error_detail)
                    Logger.error(error_detail)
                    err_count+=1
             
            #t4=time.time()
            #print(obs)
            #print(reward)
            #print(info)
            #env.render()
            #print(f"NN:{t2-t1} env: {t4-t3}")
            if done:
                if generate:
                    Logger.info("Golden created successfully")
                    pickle.dump(golden,gold)
                    exit(0)
                #obs = env.reset()
                #print(obs)
                #print(reward)
                #print(info)
                #if i == 3:
                #    reward += 1
                random.seed(seed)
                env.seed(seed)
                #tf.keras.utils.set_random_seed(seed)
                #tf.config.experimental.enable_op_determinism()
                obs=env.reset()
                #end=time.time()
                #print(end - start)
                #lh.end_iteration()
                t3=time.time()
                print(nn_exec_time/(t3-t0))
                i+=1
                if err_count !=0:
                    lh.log_error_count(int(err_count))
                lh.end_iteration()
                break


