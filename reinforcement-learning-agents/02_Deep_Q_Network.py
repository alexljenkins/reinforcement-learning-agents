import gym
import math
import random
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

class DQN(nn.Module):
    """
    A deep-q network that inherits from the torch nn.Module.
    Using the image of the cart as inputs.
    """
    def __init__(self, img_height, img_width):
        super().__init__()
        # layers, Linear = fully connected
        self.fc1 = nn.Linear(in_features = img_height*img_width*3, out_features = 60)

        # self.conv1 = nn.Conv2d(24, 62, kernel_size=5, stride=2)
        #
        # self.conv2 = nn.Conv2d(62, 24, kernel_size=5, stride=1)
        # self.bn2 = nn.BatchNorm2d(24)

        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.fc2 = nn.Linear(in_features = 60, out_features = 40)
        self.out = nn.Linear(in_features = 40, out_features = 2)

    def forward(self, t):
        """
        Required function to pass inputs and outputs along.
        """
        t = t.flatten(start_dim = 1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

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


class EpisonGreedyStrategy():
    """
    Epsilon Greedy Strategy for exploitation vs exploration Agent step choice
    """
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1.0 * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device) # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(device) # exploit

class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
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
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device = self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        """
        Returns the current state of the screen as a processed image.
        As the difference between two screens: current screen - previous screen
        """
        if self.just_starting() or self.done:
            # If initial state, 'previous state' is set to all black
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            screen1 = self.current_screen
            screen2 = self.get_processed_screen()
            self.current_screen = screen2
            return screen2 - screen1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom of screen
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8), :]
        return screen

    def transform_screen_data(self, screen):
        # convert to float, rescale and tensor convert
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40,90)),
            T.ToTensor()
        ])
        # return a batch dimension
        return resize(screen).unsqueeze(0).to(self.device)

class QValues():
    """
    No instance of the QValues class is required because we can use the
    staticmethod to directly call these functions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()

        return values


def visualize_processed_vs_unprocessed_image():
    """
    Just a quick view to ensure the image before and after processing
    are as expected.
    Reminder: we're not passing these states into the DQN, but rather
    the difference between the two images/frames.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = CartPoleEnvManager(device)
    em.reset()
    screen = em.render('rgb_array')

    plt.figure()
    plt.imshow(screen)
    plt.title('non-processed image')
    plt.show()

    screen = em.get_processed_screen()
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
    plt.title('processed image for DQN input')
    plt.show()

# visualize_processed_vs_unprocessed_image()


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
    plt.pause(0.001)                                                            # remove pause here to increase learning speed, unable to visualise graph then
    # if is_ipython:
    #     display.clear_output(wait=True)

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

# plot(np.random.rand(300), 100)

if __name__ == '__main__':
    batch_size = 256
    gamma = 0.999

    # exploration rate
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001

    # update target network every x episodes
    target_update = 1000

    memory_size = 1000
    lr = 0.001
    num_episodes = 10000

    # initialize env and agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = CartPoleEnvManager(device)
    strategy = EpisonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    # set the DQN network
    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # not in training mode
    optimizer = optim.Adam(params = policy_net.parameters(), lr=lr)

    episode_durations = []

    for episode in range(num_episodes):
        # a complete episode of agent training
        em.reset()
        state = em.get_state()

        for timestep in count():
            # action for each time step
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()

            # replay memory
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                # retrives state-action pairs for the two states
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                # loss between estimated q values and actual
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                # gradients to 0 before backprop else it would accumulate
                optimizer.zero_grad()
                # caluclates the gradients
                loss.backward()
                # updates the weights
                optimizer.step()

            if em.done:
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break

        # update the target_net weights if episode a multiple of our target update rate
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    em.close()
