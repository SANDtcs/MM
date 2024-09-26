# ****************************************************** Question 1 *************************************************************************************
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

nifty50_tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'ADANIENT.NS', 'ITC.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'LT.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'M&M.NS', 'ASIANPAINT.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'SUNPHARMA.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'BAJAJFINSV.NS', 'POWERGRID.NS', 'NTPC.NS', 'TATAMOTORS.NS', 'ONGC.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'ADANIPORTS.NS', 'HINDALCO.NS', 'BPCL.NS', 'CIPLA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'BRITANNIA.NS', 'UPL.NS', 'HEROMOTOCO.NS', 'SHREECEM.NS', 'INDUSINDBK.NS', 'SBILIFE.NS', 'BAJAJ-AUTO.NS', 'HDFCLIFE.NS', 'ICICIGI.NS', 'APOLLOHOSP.NS']
start_date = '2014-01-01'
end_date = '2024-08-01'
nifty50_data = []
for stock in nifty50_tickers:
  nifty50_data.append(yf.download(stock, start=start_date, end=end_date))

for i in range(len(nifty50_data)):
  nifty50_data[i]["Returns"] = (nifty50_data[i]["Close"] - nifty50_data[i]["Open"]) / nifty50_data[i]["Open"]

correlation_matrix = []
cov = {}
for i in range(len(nifty50_data)):
  temp_mat = []
  temp = {}
  for j in range(len(nifty50_data)):
    temp[nifty50_tickers[j]] = nifty50_data[i]['Returns'].corr(nifty50_data[j]['Returns'])
    temp_mat.append(nifty50_data[i]['Returns'].corr(nifty50_data[j]['Returns']))
  correlation_matrix.append(temp_mat)
  cov[nifty50_tickers[i]] = temp

u = np.ones(len(correlation_matrix))
w = np.matmul(u,np.linalg.inv(correlation_matrix))
w = w/np.matmul(np.matmul(u,np.linalg.inv(correlation_matrix)),np.transpose(u))


beta = {}
beta_value = 0

reliance_index = nifty50_tickers.index('RELIANCE.NS')

for i,ticker in enumerate(nifty50_tickers):
  X = list(nifty50_data[i]['Close'])
  Y = list(nifty50_data[reliance_index]['Close'])
  min_len = min(len(X), len(Y))
  X = X[:min_len]
  Y = Y[:min_len]

  beta[ticker] = np.cov(X,Y)[0][1] / np.var(Y)


for i in range(len(w)):
  beta_value += w[i] * beta[nifty50_tickers[i]]

riskFreeRate = 0.06

reliance_index = nifty50_tickers.index('RELIANCE.NS')

expReturn = riskFreeRate + beta_value * (np.mean(nifty50_data[reliance_index]['Returns']) - riskFreeRate)
print("Expected Return of the Portfolio: ", expReturn * 100, "%", sep = '')


# ****************************************************************************** Question 2 *************************************************************************************
import numpy as np
import matplotlib.pyplot as plt

def portfolioPerformance(weights, mu, cov):
  returns = np.dot(weights, mu)
  std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
  return returns, std

def generatePortfolios(n, mu, cov, rf):
  results = np.zeros((3, n))

  for i in range(n):
    weights = np.random.random(len(mu))
    weights /= np.sum(weights)
    mu_, sigma_ = portfolioPerformance(weights, mu, cov)
    results[0,i] = mu_
    results[1,i] = sigma_
    results[2,i] = (mu_ - rf) / sigma_  # Sharpe Ratio

  return results


results = generatePortfolios(1000, mu, covMatrix, 0.06)

# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], marker='o')
plt.xlabel('Portfolio Standard Deviation (Risk)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.show()

def sharpeRatio(weights, mu, cov, rf):
    returns = np.dot(weights, mu)
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return (returns - rf) / std

def maxSharpeRatio(mu, cov, rf):
    n = len(mu)
    initialWeights = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': weightConstraint})
    bounds = [(0, 1) for i in range(n)]
    result = minimize(lambda weights: -sharpeRatio(weights, mu, cov, rf), initialWeights, method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result.x

maxSharpePortfolio = maxSharpeRatio(mu, covMatrix, 0.06)
print("Max Sharpe Ratio Portfolio:", maxSharpePortfolio)

# Historical Simulation
portfolioReturns = np.dot(mvp[0], [data[ticker]['RoR'] for ticker in data])
var1 = np.percentile(portfolioReturns, 5)

# Parametric VaR
portfolioSTD = mvp[1]
var2 = portfolioReturns.mean() - 1.645 * portfolioSTD

print("Historical Simulation VaR:", var1)
print("Parametric VaR:", var2)



# Assuming you have the returns of the first two stocks in nifty50_data[0] and nifty50_data[1]
stock1_returns = nifty50_data[0]["Returns"]
stock2_returns = nifty50_data[1]["Returns"]

# Calculate the mean returns and covariance
mean_returns = np.array([np.mean(stock1_returns), np.mean(stock2_returns)])
cov_matrix = np.cov(stock1_returns, stock2_returns)

# Define the range of portfolio weights
weights = np.linspace(0, 1, 100)

# Calculate the portfolio returns and variances for different weight combinations
portfolio_returns = []
portfolio_variances = []
for w1 in weights:
  w2 = 1 - w1
  portfolio_return = w1 * mean_returns[0] + w2 * mean_returns[1]
  portfolio_variance = (w1**2 * cov_matrix[0, 0] +
                        w2**2 * cov_matrix[1, 1] +
                        2 * w1 * w2 * cov_matrix[0, 1])
  portfolio_returns.append(portfolio_return)
  portfolio_variances.append(portfolio_variance)

# Find the minimum variance portfolio
min_variance_index = np.argmin(portfolio_variances)
min_variance_return = portfolio_returns[min_variance_index]
min_variance_variance = portfolio_variances[min_variance_index]

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_variances, portfolio_returns, c='blue', marker='o', label='Portfolios')
plt.scatter(min_variance_variance, min_variance_return, c='red', marker='x', s=100, label='Minimum Variance Portfolio')
plt.xlabel('Portfolio Variance')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)
plt.show()





# ****************************************************************************** Question 3 *************************************************************************************
import matplotlib.pyplot as plt
import numpy as np

def dotMatrix(seq1, seq2, windowSize, threshold):
    matrix = np.zeros((len(seq1) - windowSize + 1, len(seq2) - windowSize + 1))
    for i in range(len(seq1) - windowSize + 1):
        for j in range(len(seq2) - windowSize + 1):
            if sum([1 for k in range(windowSize) if seq1[i + k] == seq2[j + k]]) >= threshold:
                matrix[i, j] = 1
    return matrix

seq = "AGCTTAGCTAGGCTAATCGGATCGGCTTAGCTAAGCTTAGGCT"
windowSize = 4
threshold = 3
matrix = dotMatrix(seq, seq, windowSize, threshold)

plt.imshow(matrix, cmap = 'binary')
plt.xlabel("Sequence 1")
plt.ylabel("Sequence 2")
plt.title("Dot Matrix Plot")
plt.show()

def reverseComplement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([complement[base] for base in seq[::-1]])

seq = "AGCTTAGCTAGGCTAATCGGATCGGCTTAGCTAAGCTTAGGCT"
windowSize = 4
threshold = 3
revCompSeq = reverseComplement(seq)
matrix = dotMatrix(seq, revCompSeq, windowSize, threshold)

plt.imshow(matrix, cmap = 'binary')
plt.xlabel("Sequence")
plt.ylabel("Reverse Complement Sequence")
plt.title("Dot Matrix Plot (Inverted Repeats)")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# dogSeq = "ATGCTTTTTATCTTTAACTTCTTGTTTTCCCCACTTCCAACCCCGGCACTAATCTGCATCCTGACCTTTGGAGCCGCCATCTTCCTGTGGTTGATCAATAGACCTCAGCCCGTCTTGCCTTTTGTGGATTTGGACAACCAGTCGGTGGGAATTGAGGGAGGGGCACGGAAGGGTACTGGCCAGAAGACCAATGACCCACTGTGTTACTACTACTCAGATGTCAAGACAATGTATGACGTTTTCCAAAGAGGACTTGCTGTGTCTGACAATGGGCCTTGCTTGGGATATAGAAAACCAAACCAGCCCTACAAATGGCTGTCCTACAGGCAGGTGTCTGATCGCGCAGAGTACCTGGGCTCCTGTCTCTTGCATAAAGGATATGAGCCATCATCCGACCAATCTGTTGGCATCTTTGCTCAGAATAGGCCAGAGTGGATCATCTCCGAGTTGGCTTGTTACACATACTCCATGGTAGCCGTCCCCCTGTATGACACCTTGGGAGCAGAAGCCATCATATACATTGTCAACAAGGCTGATATCGCCGCAGTGATCTGTGATACTCCCCAAAAGGCATCAACCCTGATAGAGAATATGGAGAAGGGCCTCACCCCGGGCTTGAAAATGATCATCCTCATGGATCCCTTTGAGGATGACCTGAAGGAAAGAGCAGAGAAATGTGGAATTGAGATCTTATCTCTGTTTGATGCGGAGATTCTAGGCAAAGAGAACTTCAGAAAACCTGTGCCTCCTAGACCAGAAGACCTGAGTATCATCTGCTTTACTAGTGGGACCACAGGTGACCCTAAAGGAGCCATGCTGACCCATCAAAATATTATTTCAAATGTTTCTTCTTTCCTCAAATGTATGGAGTATACTTTCAAGCCCACCCCTGAAGATGTGACCATATCCTACCTGCCCTTGGCTCATATGTTTGAGAGGATTGTACAGGCTGTTATATATTCTTGTGGTGCCAGAGTTGGTTTCTTCCAAGGAGATATTCGGTTGCTACCTGAGGACCTGAAAACTCTAAAGCCCACACTTTTTCCTTCTGTGCCTCGACTACTCAACAGGATCTATGATAAGGTACAAAATGAAGCCAAGACACCCTTGAAGAAGTTTTTATTGAACTTGGCTATTTCCTGTAAATTCAATGAAGTGAAAAAGGGTATCATCAGGCGTGACAGTATTTGGGACAAGCTCATCTTTGCAAAGATCCAGGCCACCCTTGGAGGGAGAATAAACTTTGTGGTTACTGGAGCCGCCCCCATCTCTTCTCCAGTCCTGATGTTCCTCCGGGCAGCGCTGGGATGTCCGGTGTTCGAAGCTTATGGTCAAACAGAATGCACCGCTGGCTGTACATTTACATCACCTGGGGACTGGACATCAGGGCATGTTGGAGTCCCCCTGGCTTGCAATCATGTGAAGCTAGAAGATGTACCTGACATGAACTACTTTTCAGTGAACAATGAAGGAGAGATCTGCATCAAGGGCAGCAATGTGTTCAAAGGATACCTGAAGGATCCTGAGAAAACCAAGGAAGCTCTGGATGAGGATGGCTGGCTTCACACAGGAGACATTGGTCGTTGGCTCCCGAATGGAACTCTGAAGATCATTGACCGTAAAAAGAACATTTTCAAGCTGGCCCAAGGAGAATACATTGCTCCAGAGAAGATAGAAAATATCTACATCAGGAGTAGACCAGTGTCACAAATTTTTGTGCACGGGGACAGCTTACGGTCCTCCTTAGTGGGAGTGGTGGTTCCTGACCCAGAAGTACTGCCATCATTTGTAGCCAAACTTGGGGTTAAAGGCTCCCTCGAAGAACTGTGCAAAAACAATAATGTAAGGGAAGCCATTTTAGAAGACTTGCAGAAAGTTGGGAAAGACGGTGGTCTTAAGTCCTTTGAGCAGGTCAAAAACATCTTTCTTCAACTAGAGCCATTTTCCATTGAAAATGGACTCTTGACACCAACACTGAAAGCAAAGCGGGGAGAGCTTTCCAAGTACTTTCGAACCCAAATCAACAGCCTGTATGAGAACATCCAGGAGTAG"
# dogSeq = "AGCTTAGCTAGGCTAATCGGATCGGCTTAGCTAAGCTTAGGCT"
dogSeq = "AGCGAAAGC"

def dotMatrix(seq1, seq2, windowSize, threshold):
    matrix = np.zeros((len(seq1) - windowSize + 1, len(seq2) - windowSize + 1))
    for i in range(len(seq1) - windowSize + 1):
        for j in range(len(seq2) - windowSize + 1):
            if sum([1 for k in range(windowSize) if seq1[i + k] == seq2[j + k]]) >= threshold:
                matrix[i, j] = 1
    return matrix

def reverseComplement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([complement[base] for base in seq[::-1]])

revCompSeq = reverseComplement(dogSeq)
windowSize = 1
threshold = 1

matrix1 = dotMatrix(dogSeq, dogSeq, windowSize, threshold)
matrix2 = dotMatrix(dogSeq, revCompSeq, windowSize, threshold)

fig, axes = plt.subplots(1, 2, figsize = (10, 5))

axes[0].imshow(matrix1, cmap = 'binary')
axes[0].set_xlabel("Sequence")
axes[0].set_ylabel("Sequence")
axes[0].set_xticks(list(range(9)))
axes[0].set_yticks(list(range(9)))
axes[0].set_xticklabels(list(dogSeq))
axes[0].set_yticklabels(list(dogSeq))
axes[0].set_title("Dot Matrix Plot")
axes[0].add_patch(plt.Rectangle((-0.5, -0.5), 5, 5, fill = False, edgecolor = 'red', linewidth = 2, label = 'Palindrome 1'))
axes[0].add_patch(plt.Rectangle((1.5, 1.5), 7, 7, fill = False, edgecolor = 'navy', linewidth = 2, label = 'Palindrome 2'))
axes[0].legend()

axes[1].imshow(matrix2, cmap = 'binary')
axes[1].set_xlabel("Sequence")
axes[1].set_ylabel("Reverse Complement Sequence")
axes[1].set_xticks(list(range(9)))
axes[1].set_yticks(list(range(9)))
axes[1].set_xticklabels(list(dogSeq))
axes[1].set_yticklabels(list(revCompSeq))
axes[1].set_title("Dot Matrix Plot (Inverted Repeats)")

plt.tight_layout()
plt.show()
