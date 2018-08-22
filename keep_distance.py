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

    def __init__(self,input_dim,output_dim,hidden_units,layers,learning_rate,clip_value):

        # Input
        self.input_state = tf.placeholder(tf.float32,[None,input_dim],name = "input_placeholder")

        # Network Architecture
        self.hidden_layer = tf.layers.dense(self.input_state,hidden_units,activation=tf.nn.relu)

        for n in range(1,layers):
            self.hidden_layer = tf.layers.dense(self.hidden_layer,hidden_units,activation=tf.nn.relu)

        # Network Architecture
        #self.h1 = tf.layers.dense(self.input_state,hidden_units,activation=tf.nn.relu)
        #self.h2 = tf.layers.dense(self.h1,hidden_units,activation=tf.nn.relu)
        self.output_q_predict = tf.layers.dense(self.hidden_layer,output_dim)
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

layers = 2

clip_value = 300

learning_rate = 0.001

buffer_size = 50000

batch_size = 32
update_freq = 1000
gamma = 0.99

eStart = 1
eEnd = 0.1
estep = 1000

max_train_episodes = 750
pre_train_steps = 10000 # Fill up buffer

tau = 1

dist_start = 50
ego_velo_start = 5
car_velo_start = 5

states = []
actions = []
rewards = []
rewards_time = []
reward_sum = 0
total_steps = 0

path = "./test_h32_uf_1000_e1000_50_5_5/"

path = "model_h" + str(hidden_units)+"_uf" + str(update_freq) + "_e" + str(max_train_episodes) + "_" + str(dist_start) + str(ego_velo_start) + str(car_velo_start) + "/"


#### Start training ####

tf.reset_default_graph()
mainQN = qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)
targetQN = qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateNetwork(trainables,tau)

exp_buffer = replay_buffer(buffer_size)

# Decrease randomness
epsilon = eStart
stepDrop = (eStart-eEnd)/estep

load_model = False

done = False

dist_init = dist_start
ego_velo_init = ego_velo_start
car_velo_init = car_velo_start


env = environment(dist_init,ego_velo_init,car_velo_init)

with tf.Session() as sess:
    sess.run(init)

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    for episode in range(max_train_episodes):

        #dist_init = random.random()*dist_start + 10
        #ego_velo_init = random.randint(0,ego_velo_start)
        #car_velo_init = random.randint(0,ego_velo_init)

        episodeBuffer = replay_buffer(buffer_size)
        env = environment(dist_init, ego_velo_init, car_velo_init)
        state = env.get_state()
        done = False
        reward_sum = 0

        while done == False:

            if(np.random.random() < epsilon or total_steps < pre_train_steps):
                action = np.random.randint(0,8)
            else:
                action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state]})

            state_1, reward, done = env.step(action)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([state,action,reward,state_1,done]),[1,5]))

            if total_steps > pre_train_steps:
                if epsilon > eEnd:
                    epsilon -= estep


                trainBatch = exp_buffer.sample(batch_size)

                action_max = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:np.vstack(trainBatch[:,3])}) #Action for state +1
                Qt_1_vec = sess.run(targetQN.output_q_predict, feed_dict={targetQN.input_state: np.vstack(trainBatch[:,3])}) # Qvalue for state + 1

                end_multiplier = -(trainBatch[:,4] - 1)
                Qt_1 = Qt_1_vec[range(batch_size),action_max] # Choose q_value for q + 1

                Q_gt = trainBatch[:,2] + gamma*Qt_1*end_multiplier # Qgt = reward + gamma*Q(state+1,action_max)

                ### Update the network

                _ = sess.run(mainQN.update,feed_dict={mainQN.input_state:np.vstack(trainBatch[:,0]),mainQN.q_gt:Q_gt,mainQN.actions:trainBatch[:,1]})


                if total_steps % update_freq == 0:
                    print(" Update target network")
                    updateTarget(targetOps,sess)

            reward_sum += reward
            states.append(state)
            actions.append(action)
            state = state_1
        exp_buffer.add(episodeBuffer.buffer)
        rewards_time.append(reward_sum)

        if(episode % 100) == 0:
            save_path = saver.save(sess,path+"modelRL" + str(episode)+ ".ckpt")
            print("Model saved in: %s" % save_path)

        if(episode % 10 == 0):
            print("Total steps: ",total_steps, "Average reward over 10 Episodes: ",np.mean(rewards_time[-10:]),"Epsiode: ",episode)
    saver.save(sess, path + 'model-' + str(episode) + '.ckpt')








#while done == False:
#
#    action = np.random.randint(0,8)
#    state,reward, done = env.step(action)
#    states.append(state)
#    actions.append(action)
#    rewards.append(reward)
#    reward_sum += reward
#    rewards_time.append(reward_sum)
#    t += 1
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
plt.savefig(path+'actions.png')

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
plt.savefig(path+'reward.png')
plt.show()


plt.close()










