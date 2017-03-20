import gym
import tensorflow as tf
import numpy as np
import policy
import objectives
import copy
import cv2
#This file implements the problem2 linear Q network
#utility classes are defined as follows:
class Sample:
    #This class store one sample of the tuple (s,r,a,s`,terminal)
    def __init__(self, state_in, action_in, reward_in, nextstate_in, is_terminal):
        #input states, action, reward are all lists of 4 frames since I use the
        #no frame skipping version of enduro
        self.state = state_in
        self.nextstate = nextstate_in
        self.action = action_in
        reward = sum(reward_in)
        if reward > 0:
            self.reward = 1
        elif reward == 0:
            self.reward = 0
        elif reward < 0:
            self.reward = -1

    def getProcessedState(self):
        #Flatten the current state and next state as [4*84*84] respectively:
        #We convert the uint8 data type to float32 type to let the neural
        #network better process the image
        #return: processed image as float32
        for i, oneframe in enumerate(self.state):
            tempFrame = ImagePreprocess(oneframe)
            flatten = tempFrame.flatten()
            if i == 0:
                processedState = copy.deepcopy(flatten)
            else:
                processedState = np.append(processedState, flatten)
        temp = np.asarray(processedState, dtype=np.float32)
        temp = temp.reshape([1,-1])
        return temp
    
    def getProcessedNextState(self):
        #similiar to the previous function
        #this just return the processed next state stored in the sample
        for i, oneframe in enumerate(self.nextstate):
            tempFrame = ImagePreprocess(oneframe)
            flatten = tempFrame.flatten()
            if i == 0:
                processedState = copy.deepcopy(flatten)
            else:
                processedState = np.append(processedState, flatten)
        temp = np.asarray(processedState, dtype=np.float32)
        temp = temp.reshape([1,-1])
        return temp

        
    def getSample(self):
        return (self.state, self.action, self.reward, self.nextstate)
    
    def getState(self):
        return self.state
    
    def getAction(self):
        return [np.asarray(self.action,dtype=np.int32)]
        
    def getNextState(self):
        return self.nextstate
        
    def getReward(self):
        return [np.asarray(self.reward,dtype=np.float32)]


#utility functions are defined here
def ImagePreprocess(input_image):
    #input image is either uint8 or I make it uint8 to be suitable stored on
    #memory
    #return: processed image as uint8
    beforeProcess = np.asarray(input_image, dtype = np.uint8)
    temp_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    resizedImage = cv2.resize(temp_image,(84, 84), interpolation=cv2.INTER_AREA)
    return resizedImage

def getTrainingBatch(env, batch_size, output_Q, p, sess):
    #function to generate trainning batch
    #RETURN: (stateBatch, actionBatch, rewardBatch, nextStateBatch, 
    #terminalBatch)
    #NOTE: I hard-coded the number of frames to be skipped here in this 
    #function, if you change the game from SpaceInvaders to Enduro for
    #example, you should uncomment here accordingly
    stateBatch = []
    nextStateBatch = []
    actionBatch = []
    rewardBatch = []
    terminalBatch = []
    samples = []
    for i in range(batch_size):
        if i == 0:
            action = env.action_space.sample()
            frame1, reward1, is_terminal1, info = env.step(action)
            frame2, reward2, is_terminal2, info = env.step(action)
            frame3, reward3, is_terminal3, info = env.step(action)
            # frame4, reward4, is_terminal4, info = env.step(action)
            # state = [np.copy(frame1), np.copy(frame2), np.copy(frame3), np.copy(frame4)]
            state = [np.copy(frame1), np.copy(frame2), np.copy(frame3)]
            aciton = env.action_space.sample()
            frame5, reward5, is_terminal5, info = env.step(action)
            frame6, reward6, is_terminal6, info = env.step(action)
            frame7, reward7, is_terminal7, info = env.step(action)
            # frame8, reward8, is_terminal8, info = env.step(action)
            # nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7), np.copy(frame8)]
            nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7)]
            # terminal = max([is_terminal5, is_terminal6, is_terminal7, is_terminal8])
            terminal = max([is_terminal5, is_terminal6, is_terminal7])
            # reward = [reward5, reward6, reward7, reward8]
            reward = [reward5, reward6, reward7]
            tempSample = Sample(state, action, reward,nextState, terminal)
            stateBatch.append(copy.deepcopy(tempSample.getProcessedState()[0]))
            nextStateBatch.append(copy.deepcopy(
            tempSample.getProcessedNextState()[0]))
            actionBatch.append(tempSample.getAction())
            rewardBatch.append(tempSample.getReward())
            samples.append(copy.deepcopy(tempSample))
            terminalBatch.append([float(terminal)])
            last_state = copy.deepcopy(tempSample.getProcessedState())
            if terminal == 1:
                env.reset()
        else:
            q_value = output_Q.eval(feed_dict = {x:last_state}, session = sess)
            action = p.select_action(q_value)
            frame5, reward5, is_terminal5, info = env.step(action)
            frame6, reward6, is_terminal6, info = env.step(action)
            frame7, reward7, is_terminal7, info = env.step(action)
            # frame8, reward8, is_terminal8, info = env.step(action)
            state = samples[i-1].getState()
            # nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7), np.copy(frame8)]
            nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7)]
            # terminal = max([is_terminal5, is_terminal6, is_terminal7, is_terminal8])
            terminal = max([is_terminal5, is_terminal6, is_terminal7])
            # reward = [reward5, reward6, reward7, reward8]
            reward = [reward5, reward6, reward7]

            tempSample = Sample(state, action, reward, nextState, terminal)
            stateBatch.append(copy.deepcopy(tempSample.getProcessedState()[0]))
            nextStateBatch.append(copy.deepcopy(
            tempSample.getProcessedNextState()[0]))
            actionBatch.append(tempSample.getAction())
            rewardBatch.append(tempSample.getReward())
            terminalBatch.append([float(terminal)])
            samples.append(copy.deepcopy(tempSample))
            last_state = copy.deepcopy(tempSample.getProcessedState())
            if terminal == 1:
                env.reset()
              
    return stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch


gamma = 0.99 #discount factor
alpha = 0.001 #Learning Rate
num_iteration = 1000000
test_iter = 1000#this is for me to figure out how many iteration will be
#need to achieve a resonable convergence on a trainning batch
num_update_target = 10000
batch_size = 32
env = gym.make('SpaceInvadersNoFrameskip-v0')
#if we use SpaceInvaders as suggested by the paper, we should use 3,
#while all the other games are required that we use 4
num_frame_skip = 3
env.reset()
output_num = env.action_space.n
LinearPolicy = policy.LinearDecayGreedyEpsilonPolicy(output_num, 1, 0.1, 1000000)

#define session to run:
sess = tf.Session()
#define placeholders for state, action, reward, nextstate, terminal
action = tf.placeholder(tf.int32, shape = [None,1], name = 'action')
terminal = tf.placeholder(tf.float32, shape = [None, 1], name = 'terminal')
r = tf.placeholder(tf.float32, shape = [None, 1], name = 'r')
x = tf.placeholder(tf.float32, shape = [None, num_frame_skip*84*84], name = 'x')
next_x = tf.placeholder(tf.float32, shape = [None, num_frame_skip*84*84], name = 'next_x')
online_weight = tf.Variable(tf.truncated_normal([num_frame_skip*84*84, output_num],stddev = 0.1), name = 'online_weight')
target_weight = tf.placeholder(dtype = tf.float32, shape = [num_frame_skip*84*84, output_num], name = 'target_weight')
online_bias = tf.Variable(tf.zeros([output_num]),dtype = tf.float32, name='online_bias')
target_bias = tf.placeholder(dtype = tf.float32, shape = [output_num], name='target_bias')

output_Q = tf.matmul(x, online_weight) + online_bias
target_Q = tf.matmul(x, target_weight) + target_bias

y_true = r + gamma * tf.reduce_max(target_Q) * (1-terminal)
y_pred = tf.gather(output_Q, action)

#create the loss function and set up the training step:
#samples and form a batch to calculate the loss
loss = tf.reduce_mean(objectives.huber_loss(y_true, y_pred))

train_step = tf.train.AdamOptimizer(1e-04).minimize(loss)
sess.run(tf.global_variables_initializer())
#doing the trainning loop:
for i in range(num_iteration):
    target_w = online_weight.eval(session = sess)
    target_b = online_bias.eval(session = sess)
    stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch = \
    getTrainingBatch(env, batch_size, output_Q, LinearPolicy, sess)
    feed = {x:stateBatch, action:actionBatch, r:rewardBatch, next_x:
        nextStateBatch, terminal: terminalBatch, target_weight:target_w,
        target_bias:target_b}
    rewardBatch = np.asarray(rewardBatch)
    print("Current total reward is:")
    print(np.sum(rewardBatch))
    for j in range(test_iter):
        train_step.run(feed_dict = feed,session = sess)
            
    
    
