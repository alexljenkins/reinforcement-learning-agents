import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0,"C:\\Program Files\\ffmpeg\\bin")

# hyper parameters
EPOCHS = 200  # number of episodes
EXPLORE_MAX = 1  # e-greedy threshold start value
EXPLORE_MIN = 0.05  # e-greedy threshold end value
DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.002  # NN optimizer learning rate
BATCH_SIZE = 1000  # Q-learning batch size
WATCH = 100 # display for last x tests

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, 24)
        # self.l2 = nn.Linear(50, 24)
        self.l5 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        x = self.l5(x)
        return x

def select_action(state, epoch):
    global steps_done, total_epochs
    sample = random.random()
    # exploration_threshold = EXPLORE_MIN + (EXPLORE_MAX - EXPLORE_MIN) * math.exp(-1.0 * (epoch+total_epochs) * 0.005)
    exploration_threshold = EXPLORE_MIN + ((EXPLORE_MAX - EXPLORE_MIN) * math.exp(-1. * steps_done / DECAY))
    steps_done += 1
    if sample > exploration_threshold:
        return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def run_episode(epoch, environment):
    state = environment.reset()
    steps = 0
    while True:
        if epoch > (EPOCHS-WATCH):
            environment.render()
        action = select_action(FloatTensor([state]), epoch)
        next_state, reward, done, _ = environment.step(action[0][0].item())

        reward = reward / (abs(next_state[0])+0.0000001)
        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done:
            epoch_durations.append(steps)
            if (epoch+total_epochs) % (EPOCHS-WATCH):
                plot_durations()
            break
    epoch +=1



def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values.squeeze(0), expected_q_values.unsqueeze(1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def load_trained_model(path):
    """
    Loads the weights, past epoch scores and memory for the model.
    """
    checkpoint = torch.load(path)
    print(checkpoint['epoch_durations'])
    model.load_state_dict(checkpoint['model'])
    memory.memory += checkpoint['memory']
    epoch_durations = checkpoint['epoch_durations']
    total_epochs = checkpoint['total_epochs']
    print("LOADED!")
    print(epoch_durations)
    model.eval()
    return model, memory.memory, epoch_durations, total_epochs


def save_trained_model(path):
    """
    Saves the weights, past epoch scores and memory for the model.
    """
    torch.save({
            'model': model.state_dict(),
            'memory': memory.memory,
            'epoch_durations': epoch_durations,
            'total_epochs': total_epochs
            }, path)

    print("SAVED!")

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(epoch_durations)
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    model = Network()
    if use_cuda:
        model.cuda()
    memory = ReplayMemory(50000)
    optimizer = optim.Adam(model.parameters(), LR)
    steps_done = 0
    epoch_durations = []
    total_epochs = 0
    # model, memory.memory, epoch_durations, total_epochs = load_trained_model('pretrained_models/08trained.pth')
    try:
        model, memory.memory, epoch_durations, total_epochs = load_trained_model('pretrained_models/08trained.pth')
    except:
        print("no model loaded")

    for e in range(EPOCHS):
        run_episode(e, env)
        total_epochs += 1

    save_trained_model('pretrained_models/08trained.pth')
    print('Complete')
    env.close()
    plt.ioff()
    plt.show()
