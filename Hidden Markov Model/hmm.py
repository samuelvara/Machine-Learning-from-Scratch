import json
import numpy as np
from numpy.core.fromnumeric import trace


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        #From slide 27, Alpha(s, t) = b(s, xt) Sigma ( a(s', s) * Alpha(s', t-1))
        #Alpha(s, 1) = pi(s) * b(s, x1)
        #Following Slide 28
        for s in range(S):
            alpha[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for s in range(S):
                sigma_part = 0
                for s_prime in range(S):
                    sigma_part+=self.A[s_prime][s] * alpha[s_prime][t-1]
                alpha[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * sigma_part
        return alpha


    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        #From slide 30, Beta (s, t) = Sigma( a(s, s_prime) * b(s_prime, x(t+1)) * Beta(s_prime, t+1))
        #Beta(s, T) = 1
        for s in range(S):
            beta[s][L-1] = 1
        
        for t in range(L-2, -1, -1):
            for s in range(S):
                sigma_part = 0
                for s_prime in range(S):
                    sigma_part+=self.A[s][s_prime] * self.B[s_prime][self.obs_dict[Osequence[t+1]]] * beta[s_prime][t+1]
                beta[s][t] = sigma_part
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        #From slide 31, P(X_{1:T}=x_{1:T}) = Sigma_s(Alpha(s, t) * Beta(s, t))
        # At T, Beta(s, T) = 1
        S = len(self.pi)
        L = len(Osequence)
        prob = 0
        Alpha = self.forward(Osequence)
        for s in range(S):
            prob+=Alpha[s][L-1]
        return prob
        

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        #from Slide 31, Gamma(s, t) = (Alpha(s, t) * Beta(s, t)) / P(X_{1:T}=x_{1:T})
        S = len(self.pi)
        L = len(Osequence)
        gamma = np.zeros([S, L])
        Alpha = self.forward(Osequence) 
        Beta = self.backward(Osequence)
        norm = 0 
        for s in range(S):
            norm+=Alpha[s][L-1]
        for s in range(S):
            for t in range(L):
                gamma[s][t] = (Alpha[s][t] * Beta[s][t]) / norm
        return gamma

    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        # On expanding we get prob[s, s', t-1] = (Alpha(s, t) * a(s, s') * b(s', x(t+1)) * Beta(s', t+1)) / P(X_{1:T}=x_{1:T}
        Alpha = self.forward(Osequence) 
        Beta = self.backward(Osequence)
        norm = 0 
        for s in range(S):
            norm+=Alpha[s][L-1]
        for s in range(S):
            for s_prime in range(S):
                for t in range(L-1):
                    prob[s, s_prime, t] = (Alpha[s][t] * self.A[s][s_prime] * self.B[s_prime][self.obs_dict[Osequence[t+1]]] * Beta[s_prime][t+1]) / norm
        return prob


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        L = len(Osequence)
        Delta = np.zeros([S, L])
        Triangle = np.zeros([S, L], dtype='int')
        for s in range(S):
            Delta[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        
        for t in range(1, L):
            for s in range(S):
                delta, ind = -1, -1
                for s_prime in range(S):
                    temp = self.A[s_prime][s] * Delta[s_prime][t-1]
                    if temp > delta:
                        delta = temp
                        ind = s_prime
                Delta[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * delta
                Triangle[s][t] = ind
                # Delta[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * max(self.A[s_prime][s] * Delta[s_prime][t-1] for s_prime in range(S))
                # Triangle[s][t] = np.argmax(self.A[s_prime][s] * Delta[s_prime][t-1] for s_prime in range(S))  

        z_star = [np.argmax(Delta[:, L-1])]
        for t in range(L-1, 0, -1):
            z_star.append(Triangle[z_star[len(z_star)-1]][t])
        z_star = z_star[::-1]
        
        path = [0] * len(z_star)

        for s, obs in self.state_dict.items():
            for i, z in enumerate(z_star):
                if obs == z:
                    path[i] = s
        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
