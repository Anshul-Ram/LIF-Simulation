# Appendix A: LIF model code.

# Importing relevant modules
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Defining the parameter values.
# The units for these are mV.
Tau = 20
E0 = -60
Vre = -55
Vth = -50
sigma = 4

# Calculating time steps for the simulation.
# The units for these are ms.
dt = 0.01
T = 1000

## Simulation of the LIF equation.

> Simulated
using
parameter
values:
Tau = 20
ms, E = -60
mV, Vre = -55
mV, *Vth = -50 * mV, sigma = 4
mV
with dt = 0.01 ms and T = 1000 ms.

# Simulating the LIF equation.

# This function can now be called independently, with different arguments


def Equation_2(Tau, E0, Vth, Vre, sigma, dt, T):
    time_steps = round(T / dt)
    # Defining a vector for V
    V = np.zeros(time_steps + 1)

    #  I have used V_re for the intial voltage.
    # Start at E_0, starting at V_re is the same as if the neuron just fired
    V[0] = E0

    # Keeping count of the number of firings 
    Firings = 0

    # Calculating Gaussian white noise (Xi term), I have precalculated the white noise.
    # Mean = 0, standard deviation = 1
    noise = (sigma * math.sqrt((2 * dt) / Tau)) * np.random.normal(loc=0, scale=1, size=(time_steps + 1))

    for i in range(0, time_steps):

        # Euler integration to find V
        # Change this so you use i, i + 1 instead i - 1, i.
        V[i + 1] = V[i] + (dt / Tau) * (E0 - V[i]) + noise[i]

        # Code to reset the voltage to the reset voltage and to count the number of firings
        if V[i + 1] >= Vth:
            V[i + 1] = Vre
            Firings += 1

    #  Calculating the firing rate, 'T' is in ms
    #  Consider changing this to account for the units
    # I've broken this correct this
    r = Firings / (T * (10 ** (-3)))
    # return time_steps
    return r, V, time_steps


r, V, time_steps = Equation_2(Tau, E0, Vth, Vre, sigma, dt, T)
print(f"The firing rate is r: {round(r, 2)} Hz")

#  Plotting the voltage over time
# On some occasions the plot is empty on other occassions the plot has an upward curve to it.
# Define np.linspace(0, T, time_steps + 1) as an object to use, reducing additional code
plt.figure(figsize=(14, 4))
plt.plot(np.linspace(0, T, time_steps + 1), V)
plt.plot(np.linspace(0, T, time_steps + 1), Vth * np.ones(time_steps + 1), 'r-')
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("LIF model of neuronal voltage")
plt.axis([0, T, -80, -40])

print("The blue line is the voltage and the red line is the threshold voltage")

## Finding the average firing rate.

> Running
the
LIF
model
multiple
times
for T = 10, 000s and finding the average firing rate.Tested with different values of E_0.

'''

A block of code that runs the LIF model multiple times for various values of E_0

'''

# Calculating time steps for the simulation.
# The units for these are ms.
dt = 0.01
T = 10000
time_steps = round(T / dt)

# Defining a vector for V
V = np.zeros(time_steps + 1)

average_r = 0
number_of_runs = 5

# Make explicit that this is changing the values of E_0
for i in range(-64, -55, 2):
    E0 = i
    average_r = 0
    for i in range(number_of_runs):
        r, V, time_steps = Equation_2(20, E0, -50, -55, 4, dt, T)
        average_r += r
    average_r = (average_r / number_of_runs)
    # Save these results into a vector and plot them as a vector
    # Need the ultimate output
    print(f"The value of the firing rate 'r' averaged {number_of_runs} times, T = {T} ms, with a value of E0 = {E0} mV")
    print(f"E0 = {E0} mV, average r = {round(average_r, 2)} Hz \n")

## Finding the firing rate using an approximation.


# Defining variables.
# Change these units ,be careful with these units
# * ( 10 ** (-3))
# These are in mV
Tau = 20
E0 = -60
Vth = -50
Vre = -55
sigma = 4


# Defining Equation 16.
def Equation_16(z, Tau, E0, Vth, Vre, sigma):
    # This 'if' statement is here to account for z = 0, where the integrand is not defined.
    # I have approximated the value of Equation 16 here.
    # You can find the value properly by taking the limit.
    x_th = (Vth - E0) / sigma
    x_re = (Vre - E0) / sigma

    # The integrand cannot be evaluated at z = 0, however it tends towards x_th - x_re from above.
    if z == 0:
        # If the z value is 0 then return the limit of the integrand from above at 0.
        return x_th - x_re

    return (z ** -1) * ((math.exp(x_th * z) - math.exp(x_re * z)) * math.exp((-1 / 2) * (z ** 2)))


# Trapezoidal integration
# The integrand of Eq. 16 tends to ~ 1.25 at z = 0, and becomes negligible ~ 7
def integrate_trapezoid(n, initial, final, Tau, E0, Vth, Vre, sigma):
    sum = 0

    # Finding when the function tails off towards 0.
    value = Equation_16(final, Tau, E0, Vth, Vre, sigma)

    # Checking that the integrand starts to tail at the 'final' value.
    # Analyse the integrand for large z, and find an approximation to the integrand for large z.
    '''
    This truncation criteria is odd
    '''

    while abs(value) > 0.001:
        final += 0.1
        value = Equation_16(final, Tau, E0, Vth, Vre, sigma)

    # Trapezoidal integration.
    # Write this as a function
    for i in range(n - 1):
        a = initial + i * (final - initial) / n
        b = initial + (i + 1) * (final - initial) / n
        trap = (1 / 2) * (b - a) * (Equation_16(b, Tau, E0, Vth, Vre, sigma) + Equation_16(a, Tau, E0, Vth, Vre, sigma))
        sum = sum + trap
        # (z ** -1) * (( math.exp(x_th * z) - math.exp(x_re * z) ) * math.exp( (-1/2) * (z ** 2) ))
    return sum


sum = integrate_trapezoid(20, 0, 6.6, Tau, E0, Vth, Vre, sigma)
print(
    f"The value given by the Trapezoidal rule for integration with number of Trapeziums n = 20: Z = {round(sum, 4)} \n")

# Calculating the firing rate
r = ((Tau * (10 ** (-3))) * round(sum, 4)) ** (-1)
print(f"The value of r = {round(r, 2)} Hz for  Trapezoidal rule integration")

# Trying to find out why the integrand in Eq. 16 diverges.

Tau = 20
E0 = -60
Vth = -50
Vre = -55
sigma = 4
z = 9

print(f"Integrand of Eq. 16 (z = {z}): ", Equation_16(z, Tau, E0, Vth, Vre, sigma))

## Second approximation of the firing rate.

'''
For E0 > Vth & low noise.
This only applies when xth << 0 
'''


def Equation_18(Tau, E0, Vth, Vre, sigma):
    r = 1 / ((Tau * (10 ** (-3))) * math.log((E0 - Vre) / (E0 - Vth)))

    return r


## Using equation 20 of week 8 of the notes to approximate r.


'''
For E0 < Vth
Using Eq. 20 as E_0 < V_th (subthreshold drive)

'''


def Equation_20(Tau, E0, Vth, Vre, sigma):
    x_th = (Vth - E0) / sigma
    x_re = (Vre - E0) / sigma
    if E0 < Vth and x_re > 0:
        Z = math.exp((1 / 2) * (x_th ** 2)) * math.sqrt(2 * math.pi) / x_th
        r = ((Tau * (10 ** (-3))) * Z) ** (-1)

        '''
        print(f"Tau = {Tau} ms, E_0 = {E0} mV, V_th = {Vth} mV, V_re = {Vre} mV, V_th = {Vth} mV, sigma = {sigma} mV")
        print(f"The value of x_th = {round(x_th, 2)}")
        print(f"The value of Z = {round(Z, 2)}")
        print(f"The firing rate r = {round(r, 2)} Hz")
    
        '''
    return r


## Using equation 21 of week 8 of the notes to approximate r.

'''
Approximation of firing rate using Equation 21
In subthreshold drive E0 < Vth
'''


def Equation_21(Tau, E0, Vth, Vre, sigma):
    r = (1 / (Tau * (10 ** (-3)))) * ((Vth - E0) / math.sqrt(2 * math.pi * (sigma ** 2))) * math.exp(
        (- ((Vth - E0) ** 2)) / (2 * (sigma ** 2)))
    return r


## Comparing theory and simulation for different E0 values.

'''
Code which runs through the simulation and theory for different values of E0
and plots them onto a graph.
'''

'''
The simulation should be dots

'''

# Defining variables.
# Change these units ,be careful with these units
# * ( 10 ** (-3))
# These are in mV
Tau = 20
Vth = -50
Vre = -55
sigma = 4

# Calculating time steps for the simulation
# The units for these are ms
dt = 0.01
T = 10000
time_steps = round(T / dt)

# Defining a vector for V
# Move the definition of V into the function definition of Equation_2
V = np.zeros(time_steps + 1)

E0_initial = -64
E0_final = -45
step_size = 1
number_of_runs = 5

# print( ( E0_final - E0_initial) / step_size )

r_simulation = np.zeros(int((E0_final - E0_initial) / step_size) + 1)
r_theory = np.zeros(int((E0_final - E0_initial) / step_size) + 1)
r_approximation = np.zeros(int((E0_final - E0_initial) / step_size) + 1)

# Make explicit that this is changing the values of E0
for i in range(int((E0_final - E0_initial) / step_size) + 1):
    E0 = E0_initial + i * step_size
    # print(E0)
    # A variable to store the average value of r in the simulation
    average_r = 0

    for j in range(number_of_runs):
        r, V, time_steps = Equation_2(Tau, E0, Vth, Vre, sigma, dt, T)
        # print(f"The value of r: {r} \n")
        average_r += r

    average_r_simulation = (average_r / number_of_runs)
    # print( f"The value of average_r_simulation: {average_r / number_of_runs}")
    r_simulation[i] = average_r_simulation

    # A block of code which will determine which r approximation from LN8 to use
    if E0 == Vth:
        # Do nothing.
        pass
    elif E0 < Vth:
        r_approximation[i] = Equation_21(Tau, E0, Vth, Vre, sigma)
    else:
        if (Tau * math.log((E0 - Vre) / (E0 - Vth))) != 0:
            r_approximation[i] = Equation_18(Tau, E0, Vth, Vre, sigma)

    # Calculating the theoretical value of r
    sum = integrate_trapezoid(20, 0, 9, Tau, E0, Vth, Vre, sigma)
    r_theory[i] = ((Tau * (10 ** (-3))) * round(sum, 4)) ** (-1)

'''
Make theory a line, make simulation dots
'''

# Plotting these results on a graph together
plt.figure(figsize=(14, 4))
'''
linspace is repeated here.
'''
plt.plot(np.linspace(E0_initial, E0_final, int((E0_final - E0_initial) / step_size) + 1), r_simulation, 'ro')
plt.plot(np.linspace(E0_initial, E0_final, int((E0_final - E0_initial) / step_size) + 1), r_theory)
# plt.plot( np.linspace(E0_initial, E0_final, int(( E0_final - E0_initial) / step_size ) + 1 ), r_approximation, 'g+')
plt.plot(np.linspace(-49, E0_final, int((E0_final - (-49)) / step_size) + 1), r_approximation[15:20], 'y')
plt.plot(np.linspace(E0_initial, -51, int((-51 - E0_initial) / step_size) + 1), r_approximation[0:14], 'k')
plt.xlabel("E0/mV")
plt.ylabel("Firing rate/Hz")
plt.title("Comparison of theory and simulation of the LIF neuron")
plt.axis([E0_initial, E0_final + 1, 0, math.ceil(np.max(r_theory) + 1)])

print(
    "The blue line is the theory and the red dots are the simulation values, the blue plusses are the approximation values")
print(
    "The blue line is the theory and the red dots are the simulation values, the black line is the approximation value")

# Plotting these results on a graph together
plt.figure(figsize=(14, 4))
'''
linspace is repeated here.
'''
plt.plot(np.linspace(E0_initial, E0_final, int((E0_final - E0_initial) / step_size) + 1), r_simulation, 'ro')
plt.plot(np.linspace(E0_initial, E0_final, int((E0_final - E0_initial) / step_size) + 1), r_theory)
# plt.plot( np.linspace(E0_initial, E0_final, int(( E0_final - E0_initial) / step_size ) + 1 ), r_approximation, 'k')

plt.plot(np.linspace(-49, E0_final, int((E0_final - (-49)) / step_size) + 1), r_approximation[15:20], 'k')
plt.plot(np.linspace(E0_initial, -51, int((-51 - E0_initial) / step_size) + 1), r_approximation[0:14], 'k')

plt.xlabel("E0/mV")
plt.ylabel("Firing rate/Hz")
plt.title("Comparison of theory and simulation of the LIF neuron")
plt.axis([E0_initial, E0_final + 1, 0, math.ceil(np.max(r_theory) + 1)])

print(
    "The blue line is the theory and the red dots are the simulation values, the black line is the approximation value")

print(r_approximation)

print(np.linspace(E0_initial, E0_final, int((E0_final - E0_initial) / step_size) + 1))

print(r_approximation.size)
print(r_approximation[0:14])
print(r_approximation[15:20])

print(np.linspace(E0_initial, -49, int((-49 - E0_initial) / step_size) + 1))
print(r_approximation[0:14])

print(np.linspace(-51, E0_final, int((E0_final - (-51)) / step_size) + 1))
print(r_approximation[15:20])

