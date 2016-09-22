import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import math
import argparse

from collections import namedtuple
LearningAgentState = namedtuple('LearningAgentState',
                                ['light', 'oncoming', 'left', 'next_waypoint'])

light_values = ['red', 'green']
action_values = Environment.valid_actions
next_waypoint_values = [i for i in Environment.valid_actions if i is not None]

light_to_idx = dict([(i[1], i[0]) for i in enumerate(light_values)])
action_to_idx = dict([(i[1], i[0]) for i in enumerate(action_values)])
next_waypoint_to_idx = dict([(i[1], i[0]) for i in enumerate(next_waypoint_values)])


def n_combos(n, k):
    """Helper function that calculates number of combinations given N things taken k at a time."""
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))


n_light_combos = n_combos(len(light_values), 1)
n_oncoming_combos = n_combos(len(action_values), 1)
n_left_combos = n_combos(len(action_values), 1)
n_next_waypoint_combos = n_combos(len(next_waypoint_values), 1)
n_combos = n_light_combos * n_oncoming_combos * n_left_combos * n_next_waypoint_combos


def hash_state(state):
    """Helper function that generates hash code based on state tuple."""
    assert(type(state) is LearningAgentState)
    hash_code = (light_to_idx[state.light] * (n_oncoming_combos * n_left_combos * n_next_waypoint_combos)) + \
                (action_to_idx[state.oncoming] * (n_left_combos * n_next_waypoint_combos)) + \
                (action_to_idx[state.left] * n_next_waypoint_combos) + \
                (next_waypoint_to_idx[state.next_waypoint])
    return hash_code


hash_to_state = dict()      # hash to state tuple dictionary
for light in light_values:
    for oncoming in action_values:
        for left in action_values:
            for next_waypoint in next_waypoint_values:
                learning_agent_state = LearningAgentState(light=light, oncoming=oncoming, left=left, next_waypoint=next_waypoint)
                hash_to_state[hash_state(learning_agent_state)] =learning_agent_state


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    valid_actions_select = ['mixed', 'random_only', 'best_only']

    def __init__(self, env, alpha, gamma, epsilon, trials, actions_select='mixed', out_q_table=False):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # q-table variables
        self.q_table = np.ndarray(shape=(n_combos, len(Environment.valid_actions)))
        self.q_table.fill(0)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        assert actions_select in self.valid_actions_select
        self.actions_select = actions_select

        self.previous_action = None
        self.previous_state = None
        self.previous_reward = None

        # statistics
        self.out_q_table = out_q_table
        self.trials = trials                                # total number of trials in this simulation
        self.current_trial_run = 0                          # index 1
        self.successes = 0
        self.successes_last_10_trials = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = None

        if self.out_q_table:
            print "Q-Table (Trial {}):".format(self.current_trial_run)
            self.print_q_table()

        self.current_trial_run += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = LearningAgentState(light=inputs['light'],
                                        oncoming=inputs['oncoming'],
                                        left=inputs['left'],
                                        next_waypoint=self.next_waypoint)
        state_idx = hash_state(state=self.state)

        # Select action according to your policy
        random_value = random.choice(range(0,101)) / 100.
        action = self.env.valid_actions[np.argmax(self.q_table[state_idx])]
        if self.actions_select == 'random_only' or (self.actions_select == 'mixed' and random_value < self.epsilon):
            action = random.choice([i for i in Environment.valid_actions if i is not None])
            print "LearningAgent.update(): random action taken (random value: {}, epsilon: {})".format(random_value, self.epsilon)
        action_idx = action_to_idx[action]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Update Q-Table
        if self.previous_state is not None:
            previous_state_idx = hash_state(state=self.previous_state)
            previous_action_idx = action_to_idx[self.previous_action]
            self.q_table[previous_state_idx][previous_action_idx] = \
                ((1 - self.alpha) * self.q_table[previous_state_idx][previous_action_idx]) + \
                (self.alpha * (self.previous_reward + self.gamma * self.q_table[state_idx][action_idx]))

        if self.env.agent_states[self]['location'] == self.env.agent_states[self]['destination']:
            self.successes += 1
            if self.current_trial_run > self.trials - 10:
                self.successes_last_10_trials += 1

        # preserve Q-table variables for update at next iteration
        self.previous_state = self.state
        self.previous_action = action
        self.previous_reward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, next_waypoint = {}, reward = {}".format(deadline, inputs, action, self.next_waypoint, reward)  # [debug]

    def print_q_table(self):
        for hash_code in xrange(self.q_table.shape[0]):
            data = list()
            for (action_idx, action) in enumerate(self.env.valid_actions):
                data.append('{}: {:<25}'.format(action, self.q_table[hash_code][action_idx]))
            print "{:>5}: {:<100} -> {}".format(hash_code, hash_to_state[hash_code], ', '.join(data))


def run():
    """Run the agent for a finite number of trials."""
    default_alpha = 0.2
    default_gamma = 0.2
    default_epsilon = 0.1
    default_update_delay = 0.5
    default_trials = 100
    default_actions_select = 'mixed'

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', dest='alpha', type=float, default=default_alpha, help='learning rate (default = %.1f)' % default_alpha)
    parser.add_argument('--gamma', dest='gamma', type=float, default=default_gamma, help='discount factor (default = %.1f)' % default_gamma)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=default_epsilon, help='exploration rate (default = %.1f)' % default_epsilon)
    parser.add_argument('--update-delay', dest='update_delay', type=float, default=default_update_delay, help='simulator update delay time (default = %.1f)' % default_update_delay)
    parser.add_argument('--trials', dest='trials', type=int, default=default_trials, help='number of trials (default = %d)' % default_trials)
    parser.add_argument('--no-display', dest='display', action='store_false', help='toggle no simulator display')
    parser.add_argument('--actions-select', dest='actions_select', type=str, default=default_actions_select, help='actions selection when updating learning agent (one of {})'.format(LearningAgent.valid_actions_select))
    parser.add_argument('--no-enforce-deadline', dest='enforce_deadline', action='store_false', help='toggle no deadline enforcement')
    parser.add_argument('--csv-output-only', dest='csv_output_only', action='store_true', help='toggle comma-separated output only')
    parser.add_argument('--q-table', dest='out_q_table', action='store_true', help='toggle q-table output')
    parser.set_defaults(display=True, enforce_deadline=True, csv_output_only=False, out_q_table=False)
    args = parser.parse_args()

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                       trials=args.trials, actions_select=args.actions_select,
                       out_q_table=args.out_q_table)  # create agent
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    e.set_primary_agent(a, enforce_deadline=args.enforce_deadline)  # specify agent to track

    # Now simulate it
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    sim = Simulator(e, update_delay=args.update_delay, display=args.display)

    sim.run(n_trials=args.trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    if args.csv_output_only:
        print "{},{},{},{},{}".format(args.alpha, args.gamma, args.epsilon, a.successes,
                                      a.successes_last_10_trials)
    else:
        print "Number of successful trips: {}".format(a.successes)
        print "Number of successful trips (last 10): {}".format(a.successes_last_10_trials)

    if args.out_q_table:
        print "Q-Table (Final):"
        a.print_q_table()


if __name__ == '__main__':
    run()
