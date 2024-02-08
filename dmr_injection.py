import mujoco_py
import gym
import sys
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import random
from pycoral.utils.edgetpu import make_interpreter



          

descision_dict = {
"Walker2d-v3": [0.7,	0.6,	0.81, 0.76,	0.58,	0.81],
"Hopper-v3":[-1,1,1],
"Humanoid-v3":[1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	-1,	-1],
"HalfCheetah-v3":[]
}

prob_dict = {
    "Walker2d-v3": [0.7,	0.6,	0.81, 0.76,	0.58,	0.81]
}

limit_dict = {
    "Walker2d-v3":[-10,10],
    "Hopper-v3":[-90,90],
    "Humanoid-v3":[-50,50],
    "HalfCheetah-v3":[-10,10]
}



if len(sys.argv) < 4:
    print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> <seed> <number of injections>")
    exit(0)
    #print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
else:
    env_name = sys.argv[1]
    seed=int(sys.argv[2])
    num_injections=int(sys.argv[3])

rng1 = random.Random()
first_errouneous_step = random.randint(0, 1000)

def insert_fault(output_rl):
    global rng1, first_errouneous_step

    liest_random_index = rng1.sample(range(len(output_rl)),random.randint(1,len(output_rl) ) )
    print(liest_random_index)
    wrong_array = output_rl
    for i in liest_random_index:
        if rng1.random() < prob_dict[env_name][i]:
            wrong_array[i] +=  100
        else: 
            wrong_array[i] -= 100
    return wrong_array

model_save_file = "./"+env_name+"_quant_edgetpu.tflite"
env_dmr = gym.make(env_name)
env_not_protected = gym.make(env_name)


interpreter_dmr1 = make_interpreter(model_save_file)
interpreter_dmr2 = make_interpreter(model_save_file)
interpreter_not_protected = make_interpreter(model_save_file)

interpreter_dmr1.allocate_tensors()
interpreter_dmr2.allocate_tensors()
interpreter_not_protected.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_dmr1.get_input_details()
output_details = interpreter_dmr1.get_output_details()



num_inj = 0

while num_inj < num_injections:
    step_counter_dmr = 0
    step_counter_np = 0
    done_dmr=False
    done_np=False
    random.seed(seed)
    env_not_protected.seed(seed)
    obs_np = env_not_protected.reset()

    env_dmr.seed(seed)
    obs_dmr = env_dmr.reset()
    
    for j in range(1000):
   
        if not done_dmr:
            input_dmr = tf.cast(obs_dmr.reshape(1, -1),tf.float32)
            interpreter_dmr1.set_tensor(input_details[0]['index'], input_dmr)
            interpreter_dmr2.set_tensor(input_details[0]['index'], input_dmr)
            interpreter_dmr1.invoke()
            interpreter_dmr2.invoke()
            output_data_dmr1 = interpreter_dmr1.get_tensor(output_details[0]['index'])[0].numpy()
            output_data_dmr2 = interpreter_dmr2.get_tensor(output_details[0]['index'])[0]
            if j>first_errouneous_step:
                if rng1.randint(0, 1) == 0:
                    output_data_dmr1 =  insert_fault(output_data_dmr1) 
                    output_data_dmr2 = output_data_dmr2
                else:
                    output_data_dmr1 = output_data_dmr1
                    output_data_dmr2 =  insert_fault(output_data_dmr2) 
            output_data_dmr = output_data_dmr1 # to create a array that will receive the output of the dmr selection
            for index in range( len(prob_dict[env_name])):
                if prob_dict[env_name][index] > 0.5:
                    output_data_dmr[index] = min(output_data_dmr1[index],output_data_dmr2[index])
                else:
                    output_data_dmr[index] = max(output_data_dmr1[index],output_data_dmr2[index])
            obs_dmr, reward_dmr, done_dmr, inf_dmr = env_dmr.step(tf.convert_to_tensor(output_data_dmr))
            step_counter_dmr += 1  
        if not done_np:
            input_np = tf.cast(obs_np.reshape(1, -1),tf.float32)
            interpreter_not_protected.set_tensor(input_details[0]['index'], input_np)
            interpreter_not_protected.invoke()
            output_data_not_protected = interpreter_not_protected.get_tensor(output_details[0]['index'])[0].numpy()
            output_data_not_protected=insert_fault(output_data_not_protected)  
            obs_np, reward_np, done_np, info_np = env_not_protected.step(output_data_not_protected)
            step_counter_np += 1


        if done_np and done_dmr:            
            print(f"dmr: info {inf_dmr} num_steps: {step_counter_dmr} ----- not_proteted: info: {info_np} num_steps: {step_counter_np}")
            break
    num_inj+=1
   

