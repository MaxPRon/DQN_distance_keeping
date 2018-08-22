import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random



class environment:
    def __init__(self,distance_init,ego_velo_init,car_velo_init):
        self.distance = distance_init
        self.distance_init = dist_init
        self.ego_velo = ego_velo_init
        self.car_velo = car_velo_init
        self.car_velo_init = car_velo_init
        self.ego_velo_init = ego_velo_init
        self.reward = 0
        self.done = False
        self.timestep = 0
        self.action = 4

    def get_state(self):
        return [self.distance, self.ego_velo, self.car_velo]


    def step(self,action):
        self.action = action
        self.ego_action()
        self.car_move()
        self.distance += (self.car_velo-self.ego_velo)
        self.reward_function()



        if self.timestep >= 300 or self.distance < 5 or self.distance > 300:
            self.done = True
        self.timestep += 1
        return [self.distance,self.ego_velo,self.car_velo], self.reward, self.done

    def reward_function(self):
        tolerance = 5
        if self.distance > self.distance_init + tolerance:
            self.reward = -1
        if self.distance < self.distance_init - tolerance:
            self.reward = -1
        if abs(self.distance - self.distance_init) < tolerance:
            self.reward = 1
        if self.distance < 10:
            self.reward = -5

    def car_move(self):
        self.car_velo = self.car_velo_init + 0.2*self.car_velo_init*np.sin(np.deg2rad(30*self.timestep)) +np.random.random()*0.1*self.car_velo_init


    def ego_action(self):
        if self.action == 0:
            self.ego_velo += -4
        if self.action == 1:
            self.ego_velo += -3
        if self.action == 2:
            self.ego_velo += -2
        if self.action == 3:
            self.ego_velo += -1
        if self.action == 4:
            self.ego_velo += 0
        if self.action == 5:
            self.ego_velo += 1
        if self.action == 6:
            self.ego_velo += 2
        if self.action == 7:
            self.ego_velo += 3
        if self.action == 8:
            self.ego_velo += 4


### Generate Neural network

# Define In and Outputs

class qnetwork:

    def __init__(self,input_dim,output_dim,hidden_units,learning_rate,clip_value):

        # Input/ Output
        self.input_state = tf.placeholder(tf.float32,[None,input_dim],name = "input_placeholder")

        # Network Architecture
        self.h1 = tf.layers.dense(self.input_state,hidden_units,activation=tf.nn.relu)
        self.h2 = tf.layers.dense(self.h1,hidden_units,activation=tf.nn.relu)
        self.output_q_predict = tf.layers.dense(self.h2,output_dim)
        # Clip values just in case
        self.output_q_predict = tf.clip_by_value(self.output_q_predict,-clip_value,clip_value)
        # Get action (highest q-value)
        self.action_pred = tf.argmax(self.output_q_predict,1) # second axis

        # Compute Cost/Loss
        self.actions = tf.placeholder(tf.int32,shape = [None])
        self.q_gt = tf.placeholder(tf.float32, [None]) # Q-value groundtruth
        # Encode into onehot to select q-value
        self.actions_onehot = tf.one_hot(self.actions,output_dim)

        # select single Q-value given the action
        self.q_action = tf.reduce_sum(tf.multiply(self.output_q_predict,self.actions_onehot),axis = 1)

        self.cost = tf.losses.mean_squared_error(self.q_gt,self.q_action)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.minimize(self.cost)


#### Design Replay buffer

class replay_buffer():
    def __init__(self,buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size


    def add(self,exp):
        #### Check if buffer full
        if(len(self.buffer)+ len(exp) >= self.buffer_size):
            # Remove oldest exp which is too much
            self.buffer[0:(len(exp)+ len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(exp)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5]) # state,action, reward,state_1, done


#### Helper function for target network update

def updateNetwork(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []

    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value()*tau)+ ((1-tau)*tfVars[idx+total_vars//2].value())))
    return  op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)




# Parameters
input_dim = 3

output_dim = 9

hidden_units = 32

learning_rate = 0.001

clip_value = 30


dist_init = 50
ego_velo_init = 5
car_velo_init = 5

states = []
actions = []
rewards = []
rewards_time = []
reward_sum = 0
total_steps = 0

episode_number = 1999

path = './model_h32_uf10000_e2000_5055/'


#### Start Model Execution ####

tf.reset_default_graph()
mainQN = qnetwork(input_dim,output_dim,hidden_units,learning_rate,clip_value)
targetQN = qnetwork(input_dim,output_dim,hidden_units,learning_rate,clip_value)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    done = False
    sess.run(init)
    #ckpt = tf.train.get_checkpoint_state(path)
    #saver.restore(sess,ckpt.model_checkpoint_path)
    saver.restore(sess,path+'model-' + str(episode_number) + '.ckpt')
    env = environment(dist_init,ego_velo_init,car_velo_init)
    state = env.get_state()



    while done == False:
        action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state]})
        #action = random.randint(0,8)
        state1, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        reward_sum += reward
        rewards_time.append(reward_sum)
        state = state1






states = np.asarray(states)
#rewards = np.asarray(rewards)



plt.figure(1)
ax1 = plt.subplot(3,1,1)# rows, cols, index
ax1.set_title("Distance")
ax1.set_xlabel("timestep")
ax1.set_ylabel("distance in m")
ax1.plot(states[:,0])
ax1.axhline(dist_init,color="r",linestyle ="--",linewidth = 0.5)

ax2 = plt.subplot(3,1,2)
ax2.set_title("Ego-velocity")
ax2.set_xlabel("timestep")
ax2.set_ylabel("velocity in m/s")
ax2.axhline(ego_velo_init,color="r",linestyle ="--",linewidth = 0.5)
ax2.plot(states[:,1])

ax3 = plt.subplot(3,1,3)
ax3.set_title("Car-Velocity")
ax3.set_xlabel("timestep")
ax3.set_ylabel("velocity in m/s")
ax3.axhline(car_velo_init,color="r",linestyle ="--",linewidth = 0.5)
ax3.plot(states[:,2])
plt.tight_layout()
plt.show(block=False)

plt.figure(2)
#ax = plt.subplot(2,1,1)
#ax.set_title("Reward")
#ax.set_xlabel("timestep")
#ax.set_ylabel("reward")
#ax.plot(rewards)

ax = plt.subplot(1,1,1)
ax.set_title("Reward over time")
ax.set_xlabel("timestep")
ax.set_ylabel("reward")
ax.plot(rewards_time)


plt.tight_layout()
plt.show()

plt.close()



