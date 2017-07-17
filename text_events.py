import sys

from six import StringIO

from gym import Env, spaces, utils
from gym.utils import seeding

class TextEventsEnv(Env):
    """
    Text events environment created from simulated dataset
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, dataset_path="./data/dataset.csv", start_length=5, n_lives=3):
        self.data = load_dataset(dataset_path=dataset_path)
        self.start_length = start_length
        self.possible_events = list(set.union(*[set(d) for d in self.data]))
        self.max_story_length = len(max(self.data, key=len))

        self.action_space = spaces.Discrete(len(self.possible_events))
        self.observation_space = spaces.Box(0, self.action_space.n-1, (self.max_story_length,))

        self.n_lives = n_lives

        self.rewards = {"accurate":1,
                        "error":-5,
                        "end":10,
                        "life":2}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.story = self.np_random.choice(self.data)
        self.lives = self.n_lives
        self.story_progress = self.start_length #Pointer to next event to predict
        self.colors = ["cyan"] * self.start_length # For rendering
        self.s = self.story[:self.start_length] # Environment state
        return self.s

    def _step(self, a):
        # Verifying action input
        if isinstance(a, (int, long)):
            a = self.possible_events[a]
        if not a in self.possible_events:
            raise ValueError('The action %c is not in the action space %s' \
                                                        % (a, str(self.possible_events)))

        done = False
        reward = 0
        info = {}

        expected_event = self.story[self.story_progress]

        # Processing the action in the environment
        if expected_event == a:
            reward = self.rewards["accurate"]
            self.colors.append("green")
        else:
            reward = self.rewards["error"]
            self.colors.append("red")
            self.lives -= 1
            if self.lives == 0:
                done = True
                reward -= self.rewards["end"]

        # Progressing in the story
        self.s += a
        self.story_progress += 1

        # Informations about this step in the environment
        info["true_event"] = expected_event
        info["lives"] = self.lives
        info["steps"] = self.story_progress - self.start_length

        # Detecting the end of the story
        if (self.story_progress == len(self.story)) and not done:
            done = True
            reward += self.rewards["end"] + self.lives * self.rewards["life"]

        return self.s, reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        for event, color in zip(self.s, self.colors):
            outfile.write(utils.colorize(event, color))

        # No need to return anything for human
        if mode != 'human':
            outfile.write("\n")
            return

        # Dynamic cmd line displaying
        if self.lives == 0:
            outfile.write('\n')
        else:
            outfile.write('\r')

def load_dataset(dataset_path):
    """
    Loads a text event based dataset in a csv format with one story (set of events) per row

    Parameters:
    -----------
    dataset_path: string
        Path to the dataset csv file

    Returns:
    --------
    data: list
        A list of stories (set of events) formatted in a string

    Example:
    -------
        input (csv file):
        A,B,A,A,A,B,C,D,C,D
        A,D,C,D
        B,C,C,C,C,A,A

        output (list):
        ["ABAAABCDCD", "ADCD", "BCCCCAA"]
    """
    with open(dataset_path, "r") as data_file:
        # Loading data
        data = data_file.read().split("\n")[:-1]
        # Data cleansing
        data = [d.replace(",", "").rstrip() for d in data]

        return data
