# 📘 Probability and Statistics Simulation Lab
# Author: [Your Name]
# Date: [Current Date]
# Topic: Basics of Probability, Conditional Probability, Random Variables, CLT

import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# ---------------------------------------------------
# 1. Basic Probability Simulations
# ---------------------------------------------------

# a. Tossing a coin 10,000 times
print("1(a). Coin Toss Simulation")

trials = 10000
heads = 0
tails = 0

for _ in range(trials):
    toss = random.choice(["H", "T"])
    if toss == "H":
        heads += 1
    else:
        tails += 1

p_heads = heads / trials
p_tails = tails / trials

print(f"Experimental Probability of Heads: {p_heads:.4f}")
print(f"Experimental Probability of Tails: {p_tails:.4f}\n")


# b. Rolling two dice and probability of getting a sum of 7
print("1(b). Rolling Two Dice Simulation")

dice_trials = 10000
count_sum_7 = 0

for _ in range(dice_trials):
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)
    if die1 + die2 == 7:
        count_sum_7 += 1

p_sum_7 = count_sum_7 / dice_trials
print(f"Experimental Probability of getting sum 7: {p_sum_7:.4f}\n")


# ---------------------------------------------------
# 2. Probability of getting at least one '6' in 10 rolls
# ---------------------------------------------------

print("2. Probability of getting at least one '6' in 10 rolls")

def prob_at_least_one_six(trials=10000):
    success = 0
    for _ in range(trials):
        rolls = [random.randint(1, 6) for _ in range(10)]
        if 6 in rolls:
            success += 1
    return success / trials

p_one_six = prob_at_least_one_six()
print(f"Estimated Probability: {p_one_six:.4f}\n")


# ---------------------------------------------------
# 3. Conditional Probability and Bayes' Theorem
# ---------------------------------------------------

print("3. Conditional Probability and Bayes' Theorem Simulation")

colors = ["Red"] * 5 + ["Green"] * 7 + ["Blue"] * 8
trials = 1000
data = []

# Simulate drawing with replacement
for _ in range(trials):
    first = random.choice(colors)
    second = random.choice(colors)
    data.append((first, second))

# a. P(Red | previous = Blue)
num_blue_then_red = sum(1 for f, s in data if f == "Blue" and s == "Red")
num_blue_first = sum(1 for f, _ in data if f == "Blue")

p_red_given_blue = num_blue_then_red / num_blue_first
print(f"P(Red | Blue): {p_red_given_blue:.4f}")

# b. Verify Bayes’ Theorem
# P(Blue|Red) * P(Red) / P(Blue) ≈ P(Red|Blue)

num_red_then_blue = sum(1 for f, s in data if f == "Red" and s == "Blue")
num_red_first = sum(1 for f, _ in data if f == "Red")
num_blue_first = sum(1 for f, _ in data if f == "Blue")

p_blue_given_red = num_red_then_blue / num_red_first
p_red = num_red_first / trials
p_blue = num_blue_first / trials

bayes_rhs = (p_blue_given_red * p_red) / p_blue

print(f"Bayes RHS (Expected P(Red|Blue)): {bayes_rhs:.4f}\n")


# ---------------------------------------------------
# 4. Discrete Random Variable Simulation
# ---------------------------------------------------

print("4. Discrete Random Variable Simulation")

values = [1, 2, 3]
probabilities = [0.25, 0.35, 0.40]
sample = np.random.choice(values, size=1000, p=probabilities)

mean = np.mean(sample)
variance = np.var(sample)
std_dev = np.std(sample)

print(f"Empirical Mean: {mean:.4f}")
print(f"Empirical Variance: {variance:.4f}")
print(f"Empirical Standard Deviation: {std_dev:.4f}\n")


# ---------------------------------------------------
# 5. Continuous Random Variable (Exponential Distribution)
# ---------------------------------------------------

print("5. Exponential Distribution Simulation")

mean_exp = 5
exp_samples = np.random.exponential(scale=mean_exp, size=2000)

plt.figure(figsize=(8, 5))
plt.hist(exp_samples, bins=30, density=True, alpha=0.6, label='Histogram')
x = np.linspace(0, 30, 100)
pdf = (1/mean_exp) * np.exp(-x/mean_exp)
plt.plot(x, pdf, 'r', label='PDF')
plt.title("Exponential Distribution (mean=5)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()


# ---------------------------------------------------
# 6. Central Limit Theorem Simulation
# ---------------------------------------------------

print("6. Central Limit Theorem Simulation")

# Generate uniform distribution
uniform_data = np.random.uniform(0, 1, 10000)

# Draw 1000 samples of size 30 and compute means
sample_means = [np.mean(np.random.choice(uniform_data, 30)) for _ in range(1000)]

# Plot both distributions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(uniform_data, bins=30, color='skyblue', edgecolor='black')
plt.title("Uniform Distribution")

plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=30, color='orange', edgecolor='black')
plt.title("Sample Mean Distribution (CLT)")

plt.tight_layout()
plt.show()

