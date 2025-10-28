import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# For headless environments (no GUI)
matplotlib.use("TkAgg")  # Switch to "Agg" if running on a server with no GUI

# ========== User Input ==========
ticker_1 = input("Enter first equity (e.g. AAPL): ").upper()
ticker_2 = input("Enter second equity (e.g. MSFT): ").upper()
start_date = "2023-01-01"
end_date = "2025-10-01"
# ================================

# Download price data
data = yf.download([ticker_1, ticker_2], start=start_date, end=end_date)

# Handle yfinance auto_adjust update
if 'Adj Close' in data.columns:
    data = data['Adj Close']
else:
    data = data['Close']

data = data.dropna()

# Compute daily returns
returns = data.pct_change().dropna()
r1 = returns[ticker_1].values
r2 = returns[ticker_2].values

# ======== Calculations =========
dot_product = np.dot(r1, r2)
magnitude_r2 = np.linalg.norm(r2)
scalar_proj = dot_product / magnitude_r2
vector_proj = (dot_product / (magnitude_r2 ** 2)) * r2
beta = np.cov(r1, r2)[0, 1] / np.var(r2)
corr = np.corrcoef(r1, r2)[0, 1]

# Regression Beta line
fit_line = beta * r2  # y = βx

# ======== Visualization 1: Scatter & Projection ========
plt.figure(figsize=(10, 6))
plt.scatter(r2, r1, color='gray', alpha=0.5, label=f'{ticker_1} vs {ticker_2} Returns')
plt.plot(r2, fit_line, color='blue', linewidth=2.2, label=f'Regression Line (Beta = {beta:.2f})')
plt.plot(r2, vector_proj, color='red', linewidth=1.8, label='Vector Projection')
plt.xlabel(f'{ticker_2} Daily Returns')
plt.ylabel(f'{ticker_1} Daily Returns')
plt.title(f'Relationship between {ticker_1} and {ticker_2}\n({start_date} → {end_date})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show(block=True)

# ======== Visualization 2: Histogram of Projection ========
plt.figure(figsize=(8, 4))
plt.hist(vector_proj, bins=40, color='teal', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(vector_proj), color='red', linestyle='--', linewidth=2, label='Projection Mean')
plt.xlabel('Daily Projected Return Component')
plt.ylabel('Frequency')
plt.title(f'Vector Projection Distribution of {ticker_1} on {ticker_2}')
plt.legend()
plt.tight_layout()
plt.show(block=True)

# ======== Terminal Output ========
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

print(f"\n{BOLD}================= RESULTS ================={RESET}\n")
print(f"{BOLD}Equities:{RESET} {ticker_1} vs {ticker_2}")
print(f"{BOLD}Period:{RESET}  {start_date} → {end_date}\n")

print(f"{CYAN}Scalar Projection of {ticker_1} on {ticker_2}:{RESET} {scalar_proj:,.6f}")
print(f"{CYAN}Vector Projection Length:{RESET} {len(vector_proj)}  (array not printed)")
print(f"{CYAN}Beta of {ticker_1} relative to {ticker_2}:{RESET} {beta:.4f}")
print(f"{CYAN}Correlation coefficient:{RESET} {corr:.4f}")

print(f"\n{GREEN}{BOLD}Interpretation:{RESET}")
print("- Beta > 1 → More volatile than benchmark.")
print("- Beta < 1 → Less volatile than benchmark.")
print("- Positive scalar/vector projection → Same direction movement; negative → opposite.\n")

print(f"{BOLD}===========================================\n{RESET}")
print(f"Vector Projection Mean: {np.mean(vector_proj):.6f}")
print(f"Vector Projection Std Dev: {np.std(vector_proj):.6f}")
