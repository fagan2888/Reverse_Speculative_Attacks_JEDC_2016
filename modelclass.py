# Code for "Reserve Speculative Attacks", by
# Manuel Amador, Javier Bianchi, Luigi Bocola, and Fabrizio Perri
# published at
# The Journal of Economics Dynamics and Control,
# November 2016, Volume 72, Pages 125-137
#
# Python 3.5
#
# Model class

from math import exp
from scipy.optimize import brentq
from matplotlib import pyplot as plt
import seaborn as sns


def annual_to_monthly(rate):
    return (1 + rate) ** (1/12) - 1


def monthly_to_annual(rate):
    return (1 + rate) ** (12) - 1


class Model():

    TOLERANCE = 10 ** (-7)

    def __init__(self, rbar, zlb, psi, epeg, ebar, nw, pibar, gamma, lam,
                 b0, N, g_val, c_states, c_transition, c_istar_values):
        self.psi, self.b0, self.nw, self.pibar, self.epeg, self.ebar = (
            psi, b0, nw, pibar, epeg, ebar)
        self.gamma, self.lam, self.N, self.g_val = gamma, lam, N, g_val
        self.c_states, self.c_transition, self.c_istar_values = (
            c_states, c_transition, c_istar_values)
        self.rbar, self.zlb = rbar, zlb

        # new variables
        self.b_states = range(self.N)  # b states
        self.endogenous_states = tuple((1, b, c) for b in self.b_states
                                       for c in self.c_states)
        self.states = self.endogenous_states + ((0, 0, 0), )
        # ^ (0,0,0): exogenous state
        self.transition_dict = self.create_transition_dict()

        # solving for a fixed-point
        self.e_rate = {}
        for label, e, f in (('top', self.epeg, 1), ('bottom', self.ebar, -1)):
            # Iterate from two different initial conditions (top/bottom)
            e_init = {state: e for state in self.endogenous_states}
            e_init[(0, 0, 0)] = self.ebar
            self.e_rate[label] = self.iterate(e_init, f)

        print('Difference between top and bottom iteration = {}'.format(
            self.dist(self.e_rate['top'], self.e_rate['bottom'])))

        self.i_rate, self.appreciation, self.reserves = {}, {}, {}
        self.real_money = {}
        for state in self.endogenous_states:
            exp_tmp = self.expected_e(self.e_rate['top'],
                                      self.transition_dict[state])
            e = self.e_rate['top'][state]
            self.i_rate[state] = self.r(e, exp_tmp, state)
            self.appreciation[state] = exp_tmp / e
            self.real_money[state] = self.l(self.i_rate[state],
                                            self.b_value(state))
            self.reserves[state] = self.real_money[state] + self.nw / e

    def b_value(self, state):
        """Returns the associated b_value at 'state'."""
        _, b, _ = state
        return self.b0 + self.g_val * b

    def istar_value(self, state):
        """Returns the associated istar at 'state'."""
        _, _, c = state
        return self.c_istar_values[c]

    def transition_from(self, state):
        """Returns the transition from the state 'state'. This is a vector of
        possible future states and a vector of their associated probabilities
        """
        a, b, c = state
        tomorrow_state = [(0, 0, 0)]
        if a == 0:
            proba_state = [1.0]  # exogenous state is absorbing. Done.
        else:
            proba_state = [self.lam]
            if b < self.N - 1:
                i = 1
                trans = ((1.0 - self.gamma), self.gamma)
            else:
                i = 0
                trans = (1.0, 0)
            while True:
                for cprime in self.c_states:
                    if self.c_transition[c][cprime]:
                        tomorrow_state.append((a, b + i, cprime))
                        proba_state.append((1 - self.lam) * trans[i] *
                                           self.c_transition[c][cprime])
                if i == 0:
                    break
                i -= 1
        return tomorrow_state, proba_state

    def create_transition_dict(self):
        """Returns a dictionary with all the transitions (i.e. from
        all possible states).
        """
        out = {}
        for state in self.states:
            to_states, probas = self.transition_from(state)
            out[state] = {s: p for s, p in zip(to_states, probas)}
        return out

    def expected_e(self, e_dict, transition):
        """Returns the expected exchange rate, given
        a value for the exchange rate 'e_dict' and the transition
        vector 'transition'."""
        return sum(e_dict[state] * transition[state]
                   for state in transition)

    def g(self, e, i, ist, b):
        """Central Bank loss constraint"""
        return ((1 + ist) * self.ebar / e - 1) + self.pibar / (
            e * self.l(i, b) + self.nw)

    def r(self, e, exp, state):
        """Returns the domestic interest rate given current exchange rate 'e',
        expected exchange rate value 'exp', and the state 'state'."""
        return (1 + self.istar_value(state)) * exp / e - 1

    def dist(self, e1, e2):
        """Distance between two different exchange rate mappings."""
        return max(abs(e1[state] - e2[state]) for state in
                   self.endogenous_states)

    def l(self, i, b):
        """Money demand."""
        return ((exp(b) * (self.rbar + i) ** (-self.psi)) if
                self.rbar else exp(b - self.psi * i))

    def iterate(self, e_init, from_above=0):
        """Iterates the value function and returns the exchange rate upon
        convergence. 'e_init' is a dictionary containing an exchange
        rate for all states.
        """
        counter = 0
        while True:
            e_exit = {(0, 0, 0): self.ebar}
            # distance = 0
            for state in self.endogenous_states:
                exp_temp = self.expected_e(e_init, self.transition_dict[state])
                r_temp = self.r(self.epeg, exp_temp, state)
                istar_temp = self.istar_value(state)
                b_temp = self.b_value(state)
                get_root = False
                if r_temp >= self.zlb:
                    if self.g(self.epeg, r_temp, istar_temp, b_temp) >= 0:
                        e_exit[state] = self.epeg
                    else:
                        max_e = self.epeg
                        get_root = True
                else:
                    max_e = (1 + istar_temp) * exp_temp / (1 + self.zlb)
                    if self.g(max_e, self.r(max_e, exp_temp, state),
                              istar_temp, b_temp) >= 0:
                        e_exit[state] = max_e
                    else:
                        get_root = True
                if get_root:
                    min_e = (1 + istar_temp) * self.ebar
                    e_exit[state] = brentq(
                        lambda x: self.g(x, self.r(x, exp_temp, state),
                                         istar_temp, b_temp),
                        (min_e if from_above != -1 else e_init[state] -
                            2 * self.TOLERANCE),
                        (max_e if from_above != 1 else
                         min(max_e, e_init[state] + 2 * self.TOLERANCE)),
                        xtol=self.TOLERANCE)
            distance = self.dist(e_exit, e_init)
            if not counter % 100:
                print('iter {}: distance= {}'.format(counter, distance))
            if distance < self.TOLERANCE:
                break
            e_init = e_exit
            counter += 1
        return e_exit


def do_plots(m, save_to_file=False, file_name='fig1.pdf'):
    #       sns.set_style("whitegrid")  # optional seabird settings
    sns.set_palette('Greys_d')  # optional seabird settings
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif',
                      'sans-serif': ['Computer Modern Roman']})

    b_range = list(range(4))
    markers = ('o--', 's-')
    labels = ('low $i^\star$',
              'high $i^\star$')

    fs = 15
    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    e_min = m.epeg
    for c, mk, l in zip(m.c_states, markers, labels):
        e_list = [m.e_rate['top'][(1, b, c)] for b in b_range]
        plt.plot(e_list, mk, label=l)
        e_min = min(e_min, min(e_list))
    plt.ylim([e_min * .99, m.epeg * 1.01])
    plt.title('A. Exchange Rate', fontsize=fs)
    plt.xticks(b_range, [b + 1 for b in b_range])
    plt.legend(loc=1, handlelength=4)

    plt.subplot(222)
    for c, mk, l in zip(m.c_states, markers, labels):
        plt.plot([100 * monthly_to_annual(m.i_rate[(1, b, c)])
                  for b in b_range], mk, label=l)
    plt.title('B. Monthly Interest Rate', fontsize=fs)
    plt.legend(loc=1, handlelength=4)
    plt.xticks(b_range, [b + 1 for b in b_range])
    plt.ylabel('Annualized rate, \\%')

    plt.subplot(223)
    for c, mk, l in zip(m.c_states, markers, labels):
        plt.plot([m.reserves[(1, b, c)]
                  for b in b_range], mk, label=l)
    plt.title('C. Reserves', fontsize=fs)
    plt.legend(loc=4, handlelength=4)
    plt.xlabel('b')
    plt.xticks(b_range, [b + 1 for b in b_range])
    plt.ylabel('Reserves')

    plt.subplot(224)
    plt.plot([100 * (m.reserves[(1, b, 0)] / m.reserves[(1, b, 1)] - 1)
             for b in b_range], 'o-',
             label='H to L')
    plt.title('D. Increase in Reserves: High to Low Interest', fontsize=fs)
    plt.ylabel('\\% Increase')
    plt.xlabel('b')
    plt.xticks(b_range, [b + 1 for b in b_range])

#    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    if save_to_file:
        plt.savefig(file_name)
        plt.close('all')
