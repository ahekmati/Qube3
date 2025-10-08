import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have PyQt installed
import numpy as np

# Define the function
def f(x):
    return x**2 / np.sin(x)

# Generate x values, avoiding 0 and multiples of pi to prevent division by zero
x = np.linspace(-10, 10, 1000)
x = x[np.abs(np.sin(x)) > 0.01]  # Avoid division by values too close to zero

# Compute y values
y = f(x)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x) = \frac{x^2}{\sin(x)}$', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.ylim(-100, 100)  # Limit y-range for better visualization
plt.title(r'Graph of $f(x) = \frac{x^2}{\sin(x)}$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

