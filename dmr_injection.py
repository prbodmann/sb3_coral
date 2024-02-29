import mujoco_py
import gym
import sys

import tensorflow as tf
import random




prob_dict = {
    "Walker2d-v3": [0.7,	0.6,	0.81, 0.76,	0.58,	0.81],
    "Hopper-v3":[0.21,0.8,0.79],
    "Humanoid-v3":[0.67,	0.61,	0.59,	0.60,	0.52,	0.53,	0.26,	0.47,	0.41,	0.60,	0.59,	0.25,	0.55,	0.28,	0.30,	0.10,	0.34],
    "HalfCheetah-v3":[0.31, 0.36, 0.42, 0.32,0.55,0.47]
}

limit_dict = {
    "Walker2d-v3":10,
    "Hopper-v3":90,
    "Humanoid-v3":50,
    "HalfCheetah-v3":10
}



if len(sys.argv) < 4:
    print("Usage: " + str(sys.argv[0]) + " <envname> <seed> <number of injections>")
    exit(0)
    #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
else:
    env_name = sys.argv[1]
    seed=int(sys.argv[2])
    num_injections=int(sys.argv[3])

rng1 = random.Random()
previous_selected_core = 0

def create_interpreter(env_name, cpu=False, device=":0"):
    if cpu:
        model_save_file = "./"+env_name+"_quant.tflite"
        from tensorflow import lite as tflite
        interpreter = tflite.Interpreter(model_save_file)
    else:
        model_save_file = "./"+env_name+"_quant_edgetpu.tflite"
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_save_file, device=device)

    return interpreter

def insert_fault(output_rl):
    global rng1, first_errouneous_step
    #return output_rl
    liest_random_index = rng1.sample(range(len(output_rl)),rng1.randint(1,len(output_rl) ) )
    #print(liest_random_index)
    wrong_array = output_rl
    for i in liest_random_index:
        error_mag = rng1.uniform(0,limit_dict[env_name])
        error_dist = rng1.random()
        #print(error_dist)
        if error_dist < prob_dict[env_name][i]:
            #print("sum")
            wrong_array[i] +=  error_mag
        else:
            #print("sub") 
            wrong_array[i] -= error_mag
    return wrong_array

def select_copy(output_data_dmr1,output_data_dmr2):
    global previous_selected_core
    count_0=0
    count_1=0
    for index in range( len(prob_dict[env_name])):
        #print (output_data_dmr1)
        #print (output_data_dmr2)
        #output_data_dmr[index] = (output_data_dmr1[index] + output_data_dmr2[index])/2
        #print (output_data_dmr)
        if prob_dict[env_name][index] > 0.5:
            #output_data_dmr[index] = min(output_data_dmr1[index],output_data_dmr2[index])
            if output_data_dmr1[index] < output_data_dmr2[index]:
                count_0 += 1
            elif output_data_dmr1[index] > output_data_dmr2[index]:
                count_1 += 1
            else:
                count_0 +=1 
                count_1 +=1
            
        else:
            #output_data_dmr[index] = max(output_data_dmr1[index],output_data_dmr2[index])
            if output_data_dmr1[index] > output_data_dmr2[index]:
                count_0 += 1
            elif output_data_dmr1[index] < output_data_dmr2[index]:
                count_1 += 1
            else:
                count_0 +=1 
                count_1 +=1
    if  count_0 > count_1:
        #print(f"worng core: {select_core}, core selected: core 0")
        output_data_dmr = output_data_dmr1
        previous_selected_core = 0
    elif count_0 < count_1:
        #print(f"worng core: {select_core}, core selected: core 1")
        output_data_dmr = output_data_dmr2
        previous_selected_core = 1
    else:
        #if j>first_errouneous_step:
        #    print(f"dmr 0 {output_data_dmr1 - output_data_dmr2}")
        #input()
        if previous_selected_core == 0:
            #print(f"worng core: {select_core}, core selected: core 0 equal")
            output_data_dmr = output_data_dmr1
        else:
            #print(f"worng core: {select_core}, core selected: core 1 equal")
            output_data_dmr = output_data_dmr2
    count_0=0
    count_1=0
    for index in range( len(prob_dict[env_name])):
        #print (output_data_dmr1)
        #print (output_data_dmr2)
        #output_data_dmr[index] = (output_data_dmr1[index] + output_data_dmr2[index])/2
        #print (output_data_dmr)
        if prob_dict[env_name][index] > 0.5:
            #output_data_dmr[index] = min(output_data_dmr1[index],output_data_dmr2[index])
            if output_data_dmr1[index] < output_data_dmr2[index]:
                count_0 += 1
            elif output_data_dmr1[index] > output_data_dmr2[index]:
                count_1 += 1
            else:
                count_0 +=1 
                count_1 +=1
            
        else:
            #output_data_dmr[index] = max(output_data_dmr1[index],output_data_dmr2[index])
            if output_data_dmr1[index] > output_data_dmr2[index]:
                count_0 += 1
            elif output_data_dmr1[index] < output_data_dmr2[index]:
                count_1 += 1
            else:
                count_0 +=1 
                count_1 +=1
    if  count_0 > count_1:
        #print(f"worng core: {select_core}, core selected: core 0")
        output_data_dmr = output_data_dmr1
        previous_selected_core = 0
    elif count_0 < count_1:
        #print(f"worng core: {select_core}, core selected: core 1")
        output_data_dmr = output_data_dmr2
        previous_selected_core = 1
    else:
        #if j>first_errouneous_step:
        #    print(f"dmr 0 {output_data_dmr1 - output_data_dmr2}")
        #input()
        if previous_selected_core == 0:
            #print(f"worng core: {select_core}, core selected: core 0 equal")
            output_data_dmr = output_data_dmr1
        else:
            #print(f"worng core: {select_core}, core selected: core 1 equal")
            output_data_dmr = output_data_dmr2
    return output_data_dmr
env_dmr = gym.make(env_name)
env_not_protected = gym.make(env_name)


interpreter_dmr1 = create_interpreter(env_name,cpu=True)
interpreter_dmr2 = create_interpreter(env_name,cpu=True)
interpreter_not_protected = create_interpreter(env_name,cpu=True)

interpreter_dmr1.allocate_tensors()
interpreter_dmr2.allocate_tensors()
interpreter_not_protected.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_dmr1.get_input_details()
output_details = interpreter_dmr1.get_output_details()



num_inj = 0

while num_inj < num_injections:
    first_errouneous_step = rng1.randint(0, 511)
    step_counter_dmr = 0
    step_counter_np = 0
    done_dmr=False
    done_np=False
    random.seed(0)

    #env_not_protected.seed(seed)
    obs_np, _ = env_not_protected.reset(seed=seed)

    #env_dmr.seed(seed)
    obs_dmr, _= env_dmr.reset(seed=seed)

    select_core=rng1.randint(0, 1)
    output_data_dmr = [0]*len(prob_dict[env_name])
    for j in range(1000):
        previous_selected_core = rng1.randint(0, 1)

            
        if not done_dmr:
            #print(obs_dmr)
            input_dmr = tf.cast(obs_dmr.reshape(1, -1),tf.float32)
            interpreter_dmr1.set_tensor(input_details[0]['index'], input_dmr)
            interpreter_dmr2.set_tensor(input_details[0]['index'], input_dmr)
            interpreter_dmr1.invoke()
            interpreter_dmr2.invoke()
            output_data_dmr1 = interpreter_dmr1.get_tensor(output_details[0]['index'])[0]
            output_data_dmr2 = interpreter_dmr2.get_tensor(output_details[0]['index'])[0]
            if j>first_errouneous_step:
                if  select_core== 0:
                    output_data_dmr1 =  insert_fault(output_data_dmr1) 
                    output_data_dmr2 = output_data_dmr2
                else:
                    output_data_dmr1 = output_data_dmr1
                    output_data_dmr2 =  insert_fault(output_data_dmr2)
            #print(output_data_dmr1 - output_data_dmr2)

             # to create a array that will receive the output of the dmr selection
            output_data_dmr=select_copy(output_data_dmr1,output_data_dmr2)
            obs_dmr, reward_dmr, done_dmr, someting, inf_dmr = env_dmr.step(tf.convert_to_tensor(output_data_dmr))
            step_counter_dmr += 1  
        if not done_np:
            input_np = tf.cast(obs_np.reshape(1, -1),tf.float32)
            interpreter_not_protected.set_tensor(input_details[0]['index'], input_np)
            interpreter_not_protected.invoke()
            output_data_not_protected = interpreter_not_protected.get_tensor(output_details[0]['index'])[0]
            if j>first_errouneous_step:
                output_data_not_protected=insert_fault(output_data_not_protected)  
            obs_np, reward_np, done_np, someting,info_np = env_not_protected.step(output_data_not_protected)
            step_counter_np += 1


        if done_np and done_dmr:            
            print(f"{first_errouneous_step} {step_counter_dmr} {step_counter_np} ")
            print(f"{obs_dmr} {obs_np} ")
            break
    num_inj+=1
   

