import torch
from env import Robot_Gridworld
from deep_q_learning import DeepQLearning
import matplotlib.pyplot as plt

returns_per_episode = []
gamma = 0.99
def update():
    step = 0

    for episode in range(300):

        state = env.reset()
        step_count = 0
        returns = []
        
        while True:

            env.render()
            state = torch.FloatTensor([state])
            action = dqn.choose_action(state)
            next_state, reward, terminal = env.step(action)

            step_count += 1
            returns.append(reward)

            dqn.store_transition(state, action, reward, next_state)

            if (step > 200) and (step % 5 == 0):
                dqn.learn()
        
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state
   
            if terminal == True:
                returns_per_episode.append(sum(returns))
                print(" {} End. Total steps : {}, Total returns : {}\n".format(episode + 1, step_count, returns_per_episode[episode]))
                break

            step += 1
            
   ############# Implement the codes to plot 'returns per episode' ####################
   ############# You don't need to place your plotting code right here ################

    print('Game over.\n')
    env.destroy()

if __name__ == "__main__":

    env = Robot_Gridworld()

    ###### Recommended hyper-parameters. You can change if you want ###############
    dqn = DeepQLearning(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.9,
                        e_greedy=0.05,
                        replace_target_iter=200,
                        memory_size=2000
                        )


    env.after(100, update) #Basic module in tkinter
    env.mainloop() #Basic module in tkinter


plt.plot(list(range(1,301)), returns_per_episode) 
plt.title('Returns per Episode_env')
plt.xlabel('Episode')
plt.ylabel('Returns')

