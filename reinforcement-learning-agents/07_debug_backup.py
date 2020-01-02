import gym
import math
import random
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import csv
# from torchvision.utils import save_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """
    A deep-q network that inherits from the torch nn.Module.
    Using the image of the cart as inputs.
    """
    def __init__(self, img_height, img_width):
        super(DQN, self).__init__()
        # layers, Linear = fully connected  #
        self.fc1 = nn.Linear(in_features = img_height*img_width*3, out_features = 128)

        self.fc2 = nn.Linear(in_features = 128, out_features = 256)
        self.out = nn.Linear(in_features = 256, out_features = 2)

    def forward(self, x):
        """
        Required function to pass inputs and outputs along.
        """
        x = x.flatten(start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

Experience = namedtuple('Experience',('state','action','next_state','reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        """
        Adds a new experience to memory if memory not full.
        Else pushes the memory to the front, dropping the old memory.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        """
        Selects a random sample of memories of batch_size (from memory)
        to use in the next state model.
        """
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        """
        Returns a bool if we can sample from memory.
        (checking that batch_size <= memory)
        """
        return len(self.memory) >= batch_size


class Agent():
    def __init__(self, strat_vals, num_actions):
        self.num_actions = num_actions
        self.strat_vals = strat_vals

    def get_exploration_rate(self, epoch):
        # print(self.strat_vals[1] + (self.strat_vals[0] - self.strat_vals[1]) * math.exp(-1.0 * epoch * self.strat_vals[2]))
        return self.strat_vals[1] + (self.strat_vals[0] - self.strat_vals[1]) * math.exp(-1.0 * epoch * self.strat_vals[2])

    def select_action(self, state, policy_net, epoch):
        rate = self.get_exploration_rate(epoch)

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(DEVICE) # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(DEVICE) # exploit

class CartPoleEnvManager():
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):

        return self.env.render('rgb_array')

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device = DEVICE)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        """
        Returns the current state of the screen as a processed image.
        As the difference between two screens: current screen - previous screen
        """
        if self.just_starting() or self.done:
            print("FIRST!")
            # If initial state, 'previous state' is set to all black
            self.current_screen = np.array(self.get_processed_screen())
            self.current_screen = torch.from_numpy(self.current_screen)
            # black_screen = np.zeros(self.current_screen.shape)
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            print("Second")
            screen1 = self.current_screen
            screen2 = np.array(self.get_processed_screen())
            # screen2 = torch.from_numpy(self.current_screen)
            self.current_screen = screen2
            print(f'get_state: {screen2.max()}')
            return screen2# - screen1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        print(f'get_processed_screen: {screen.max()}')
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        # Strip off top and bottom of screen
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8), :]

        # black_pixels = np.array(np.where(screen == 0))
        # first_black = black_pixels[:,0]
        # screen = screen[:,:, first_black[2]-5:first_black[2]+55]

        # print(f'0 element: {screen[0].shape}')
        # print(f'1 element: {screen[1].shape}')
        # print(f'2 element: {screen[2].shape}')
        print(f'crop_screen: {screen.max()}')
        return screen

    def transform_screen_data(self, screen):
        # convert to float, rescale and tensor convert
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
        screen = torch.from_numpy(screen)


        # use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            # T.Resize((40,90)),                                                # downsample if too computationally hard
            T.ToTensor()
        ])
        # return a batch dimension
        print(f'transform_screen_data: {screen.max()}')
        return resize(screen).unsqueeze(0).to(DEVICE)

class QValues():
    """
    No instance of the QValues class is required because we can use the
    staticmethod to directly call these functions.
    """
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(DEVICE)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()

        return values


def visualize_processed_vs_unprocessed_image():
    """
    Just a quick view to ensure the image before and after processing
    are as expected.
    Reminder: we're not passing these states into the DQN, but rather
    the difference between the two images/frames.
    """
    em = CartPoleEnvManager()
    em.reset()
    screen = em.render('rgb_array')

    # plt.figure()
    # plt.imshow(screen)
    # plt.title('non-processed image')
    # plt.show()

    screen = em.get_processed_screen()
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
    plt.title('processed image for DQN input')
    plt.show()
    # return screen
# visualize_processed_vs_unprocessed_image()

#%%
def visualize(screen):
    screen = (screen * 255)
    plt.figure()
    print(screen[0].permute(1,2,0).shape)
    print("SHAPE")
    plt.imshow(screen[0].permute(1,2,0), interpolation='none')
    plt.title('processed image for DQN input')
    plt.show()
# visualize_processed_vs_unprocessed_image()

#%%
def plot(values, moving_avg_period):
    """
    Calculates and plots the 100 step moving average for the Agent's
    survival duration.
    """
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension = 0, size = period, step = 1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))

        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def extract_tensors(experiences):
    """
    For a given batch of experiences, extracts the state, action, reward and
    next state tensors and places them in a tuple.
    """
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)


class Simulation():
    """
    Pulls together the environment, the agent, weights and actions.
    """
    def __init__(self):

        self.initialize_static_values()
        self.initialize_playground()

        if os.path.isfile('pretrained_models/cartpole.pth'):
            self.load_trained_model('pretrained_models/cartpole.pth')
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(params = self.policy_net.parameters(), lr=self.lr)
        self.target_net.eval()

    def initialize_static_values(self):
        """
        Inital values that will not updated/change if a
        model is loaded from another source.
        """
        self.memory_size = 5000 # full sample of epochs remembered
        self.batch_size = 200 # random sample drawn from for a single epoch
        self.gamma = 0.999

        # exploration rate
        # [exploration rate at start (max), min exploration rate, rate of decay]
        self.strat_vals = [1, 0.01, 0.001]

        self.target_update = 10 # update target network every x epochs

        self.lr = 0.001 # learning rate
        self.epochs = 1000 # epochs before saving and quitting


    def initialize_playground(self):
        """
        Inital values for env and agent. These values are those that will
        be updated if a model is loaded from another source.
        """
        self.epoch = 0
        self.epoch_durations = []
        self.em = CartPoleEnvManager()
        self.agent = Agent(self.strat_vals, self.em.num_actions_available())
        self.memory = ReplayMemory(self.memory_size)

        self.policy_net = DQN(self.em.get_screen_height(), self.em.get_screen_width()).to(DEVICE)
        self.target_net = DQN(self.em.get_screen_height(), self.em.get_screen_width()).to(DEVICE)


    def load_trained_model(self, path):
        """
        Loads the weights, past epoch scores and memory for the model.
        """
        checkpoint = torch.load(path)

        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        # print(self.policy_net)
        self.memory.memory += checkpoint['memory']
        # print(self.memory.memory)
        self.epoch_durations = checkpoint['epoch_durations']
        self.epoch = len(self.epoch_durations)


    def save_trained_model(self, path):
        """
        Saves the weights, past epoch scores and memory for the model.
        """
        torch.save({
                'model_state_dict': self.policy_net.state_dict(),
                'memory': self.memory.memory,
                'epoch_durations': self.epoch_durations
                }, path)

    def sim_epoch(self):
        """
        Runs the simulation for a single epoch
        """
        # a complete epoch of agent training
        self.em.reset()
        state = self.em.get_state()

        for timestep in count():
            print(f'SIM EPOCH STATE: {state.shape}')
            # action for each time step
            self.sim_timestep(state)
            print(f'sim_epoch: {state.max()}')    ###################
             #################
            visualize(state)
            # img = state[0]
            # save_image(img, f'img{timestep}.png')

            if self.em.done:
                self.epoch_durations.append(timestep)
                plot(self.epoch_durations, 100)
                break

        # update the target_net weights if epoch a multiple of our target update rate
        if self.epoch % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epoch % self.epochs == 0 and self.epoch > 10:
            self.save_trained_model('pretrained_models/cartpole.pth')
            self.em.close()


        self.epoch += 1


    def sim_timestep(self, state):
        """
        Runs a action/step for the agent within an epoch
        """
        print(f'SHAPE1(sim_timestep): {state.max()}')
        action = self.agent.select_action(state, self.policy_net, self.epoch)
        print(f'SHAPE2(sim_timestep): {state.max()}')
        reward = self.em.take_action(action)
        print(f'SHAPE3(sim_timestep): {state.max()}')
        next_state = self.em.get_state()
        print(f'SHAPE4(sim_timestep): {state.max()}')


        # replay memory
        self.memory.push(Experience(state, action, next_state, reward))
        state = next_state
        self.em.current_state = state
        print(f'SHAPE: {state.shape}')

        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            # retrives state-action pairs for the two states
            current_q_values = QValues.get_current(self.policy_net, states, actions)
            next_q_values = QValues.get_next(self.target_net, next_states)
            target_q_values = (next_q_values * self.gamma) + rewards

            # loss between estimated q values and actual
            self.loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            # print(self.loss)
            # gradients to 0 before backprop else it would accumulate
            self.optimizer.zero_grad()
            # caluclates the gradients
            self.loss.backward()
            # print(self.loss)
            # updates the weights
            self.optimizer.step()


if __name__ == '__main__':

    sim = Simulation()

    for epoch in range(sim.epoch, sim.epoch+sim.epochs+1):
        # a complete epoch of agent training
        sim.sim_epoch()

    sim.em.close()
