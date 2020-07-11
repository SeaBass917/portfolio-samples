# ================================= #
# Reciever Operating Characteristic #
# S e a B a s s                     #
# 2 0 1 9                           #
#                                   #
# Plotting the ROC for a simulated  #
# radar system.                     #
# ================================= #
# - Radar system -
# System sends pulse out and recieves the pulse reflection
# if (pulse large) : it hit plane
# else             : no plane
#
# H0 - Absense of Plane
# H1 - Presence of Plane
# X  - Signal recieved
# f(X|H0) and f(X|H1) are known
#
# <something about P(H1)>
#
# X follows the same distribution under both hypotheses, only mean value changes
#
# E[X|H0] < E[X|H1]
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Indicator function
def I(x, T):
    if(x < T):
        return 0.0
    else:
        return 1.0

def SNR(Ps, Pn):
	return 10 * np.log(Ps/Pn)

def norm(x, mean, variance):
	return ( 1/np.sqrt(2*np.pi*variance) )*np.exp(-(x-mean)*(x-mean) / (2 * variance)) 

def getVarFromSNR(snr):
	return np.power(10, -(snr/10))

# analytically determine the probability of an event
def true_p(mean, variance, T):
    F_x0 = stats.norm(mean, variance).cdf(T)
    return 1.0 - F_x0

# analytically calculate the true probability of: detection and false alarm
# for a given SNR and decision boundary
def getProbabilities(snr, T):

    variance = getVarFromSNR(snr)

    P_d = true_p(1, variance, T)
    P_fa = true_p(0, variance, T)

    return (P_d, P_fa)

# Run Monte Carlo for a given SNR
# Return (P_d, P_fa)
def run_MC(snr, T, epsilon=1e-4, k=1000, N_max=100000):

    # Determine variance
    variance = getVarFromSNR(snr)

    # monitor deltas between runs
    # mark convergence when a delta drops below epsilon k times in a row
    convergence_d_count = 0
    convergence_fa_count = 0

    # Run until convergence
    # track number of detections and false alarms
    N = N_max
    d_count = 0.0
    fa_count = 0.0
    P_d = 0.0
    P_fa = 0.0
    delta_d = 0.0
    delta_fa = 0.0
    i = 1.0
    while(i < N_max and (convergence_fa_count <= k or convergence_d_count <= k)):

        # Save the probabilities from prev run for convergence check
        P_d_last = P_d
        P_fa_last = P_fa

        # Sample
        x_i_H0 = np.random.normal(loc=0.0, scale=variance)
        x_i_H1 = np.random.normal(loc=1.0, scale=variance)

        # Update counts
        fa_count += I(x_i_H0, T)
        d_count += I(x_i_H1, T)

        # Update probabilities with new samples
        P_d = d_count / i
        P_fa = fa_count / i

        # P_d convergence check
        if(convergence_d_count <= k):
            delta_d = abs(P_d_last - P_d)
            if(delta_d < epsilon):
                convergence_d_count += 1
            else:
                convergence_d_count = 0

        # P_fa convergence check
        if(convergence_fa_count <= k):
            delta_fa = abs(P_fa_last - P_fa)
            if(delta_fa < epsilon):
                convergence_fa_count += 1
            else:
                convergence_fa_count = 0

        # increment
        i+=1.0

    return (P_d, P_fa)

# Run the main experiment
def experiment():

    # Get range of snr
    range_SNR = [5, 4, 3, 2, 1, 0, -1]

    # Colors at each SNR
    color_by_snr = ["#f0f921","#fdb42f","#ed7953","#cc4778","#9c179e","#5c01a6","#0d0887"]

    # Range of T's
    range_T = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # at each snr determine (P_d, P_fa) for each threshold value
    for snr, color in zip(range_SNR, color_by_snr):

        # Get the ROC for each T
        ROC = [run_MC(snr, T, epsilon=1e-4, k=10000, N_max=100000) for T in range_T]

        # Break them into seperate arrays for ploting
        y = []
        x = []
        for (P_d, P_fa) in ROC:
            y.append(P_d)
            x.append(P_fa)

        # Plot the ROC
        label = str(snr)+' SNR'
        plt.plot(x, y, color=color, marker='o', label=label)
        for i, T in enumerate(range_T):
            plt.annotate('T='+str(T), xy=(x[i], y[i]), xytext=(x[i] - 0.0175, y[i] + 0.001))
            # plt.annotate(str(T), xy=(x[i], y[i]), xytext=(x[i] - 0.01, y[i] + 0.001))

    plt.title('ROC Between -1 and 5 SNR', fontsize=24)
    plt.xlabel('Probability of False Alarm', fontsize=18)
    plt.ylabel('Probability of Detection', fontsize=18)
    plt.legend(loc='upper left')
    plt.show()	

# Use the analytical probabilities to determine error in the MC experiment
def determine_error():

    # Get range of snr
    range_SNR = [5, 4, 3, 2, 1, 0, -1]

    # Colors at each SNR
    color_by_snr = ["#f0f921","#fdb42f","#ed7953","#cc4778","#9c179e","#5c01a6","#0d0887"]

    # Range of T's
    range_T = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # at each snr determine (P_d, P_fa) for each threshold value
    for snr, color in zip(range_SNR, color_by_snr):

        # Get the ROC for each T
        ROC = [run_MC(snr, T, epsilon=1e-4, k=10000, N_max=100000) for T in range_T]
        ROC_a = [getProbabilities(snr, T) for T in range_T]

        # calculate error for both probabilities and average them at each element
        err = [(abs(P_d - P_d_a) + abs(P_fa - P_fa_a))/2.0 for ((P_d, P_fa),(P_d_a, P_fa_a)) in zip(ROC, ROC_a)]

        # Plot the ROC
        label = str(snr)+' SNR'
        plt.plot(range_T, err, color=color, marker='o', label=label)

    plt.title('Errors', fontsize=24)
    plt.xlabel('T', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.legend(loc='upper left')
    plt.show()	

# Plot both distributions at a given SNR
def plot_H0_H1(snr):

    variance = getVarFromSNR(snr)

    res = 1e-1

    x = np.arange(-3, 4, res)

    y_H0 = [norm(x_i, 0, variance) for x_i in x]
    y_H1 = [norm(x_i, 1, variance) for x_i in x]

    plt.plot(x, y_H0, c='red')
    plt.plot(x, y_H1, c='blue')

    # vertical line
    x_vert = [0.5 for i in range(700)]
    y_vert = np.arange(0, 0.7, 0.001)
    plt.plot(x_vert, y_vert, c='black')

# Script to plot the two cirves at the min and max SNR
def compare_curves():

    plt.figure(0)
    plot_H0_H1(-1)
    plt.figure(1)
    plot_H0_H1(5)
    plt.show()

# Same curve but calculate true probabilities rather then experimentally deriving them
def analytical_experiment():

    # Get range of snr
    range_SNR = [5, 4, 3, 2, 1, 0, -1]

    # Colors at each SNR
    color_by_snr = ["#f0f921","#fdb42f","#ed7953","#cc4778","#9c179e","#5c01a6","#0d0887"]

    # Range of T's
    range_T = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # at each snr determine (P_d, P_fa) for each threshold value
    for snr, color in zip(range_SNR, color_by_snr):

        # Get the ROC for each T
        ROC = [getProbabilities(snr, T) for T in range_T]

        # Break them into seperate arrays for ploting
        y = []
        x = []
        for (P_d, P_fa) in ROC:
            y.append(P_d)
            x.append(P_fa)

        # Plot the ROC
        label = str(snr)+' SNR'
        plt.plot(x, y, color=color, marker='o', label=label)
        for i, T in enumerate(range_T):
            plt.annotate('T='+str(T), xy=(x[i], y[i]), xytext=(x[i] - 0.0175, y[i] + 0.001))
            # plt.annotate(str(T), xy=(x[i], y[i]), xytext=(x[i] - 0.01, y[i] + 0.001))

    plt.title('ROC Between -1 and 5 SNR', fontsize=24)
    plt.xlabel('Probability of False Alarm', fontsize=18)
    plt.ylabel('Probability of Detection', fontsize=18)
    plt.legend(loc='upper left')
    plt.show()	

if __name__ == '__main__':
    experiment()
    # compare_curves()
    # analytical_experiment()
    # determine_error()