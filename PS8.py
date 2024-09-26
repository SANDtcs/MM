import pandas as pd
from math import sqrt
import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have seq1 defined as in the previous code

def reverse_complement(seq):
  """Calculates the reverse complement of a DNA sequence."""
  complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
  return ''.join([complement[base] for base in seq[::-1]])

def dot_matrix_self_comparison(seq, window, threshold):
  """Performs self-comparison dot matrix analysis and detects palindromes."""
  seq_rc = reverse_complement(seq)
  matrix = np.zeros((len(seq)-window+1, len(seq)-window+1))

  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      if np.sum([1 for k in range(window) if seq[i+k] == seq[j+k]]) >= threshold:
        matrix[i][j] = 1

  return matrix

def find_palindromes(matrix, seq, window):
  """Identifies palindromic sequences from the dot matrix."""
  palindromes = []
  for i in range(len(matrix)):
    for j in range(i + window, len(matrix)):  # Consider only upper triangle
      if matrix[i][j] == 1:
        if i + window <= j:
            palindrome = seq[i:j]
            palindromes.append((i, j, palindrome))
  return palindromes


matrix = dot_matrix_self_comparison(seq1, windowSize, threshold)

plt.figure(figsize=(10, 10))
plt.imshow(matrix, cmap='binary', interpolation='nearest')
plt.title('Dot Matrix Self-Comparison')
plt.xlabel('Sequence Position')
plt.ylabel('Sequence Position')

# Find and annotate palindromes
palindromes = find_palindromes(matrix, seq1, windowSize)
for start, end, palindrome in palindromes:
  plt.plot([start, end], [end, start], 'r-', linewidth=1)


plt.show()


# *****************************************************************************************************************************************************************************************************************************
import pandas as pd
from math import sqrt
import os
import matplotlib.pyplot as plt
import numpy as np
import time


def optimized_dot_matrix(seq1, seq2, window, threshold):
  """Optimized dot matrix computation using a sliding window approach."""
  matrix = np.zeros((len(seq1)-window+1, len(seq2)-window+1))

  for i in range(len(seq1)-window+1):
    for j in range(len(seq2)-window+1):
      match_count = 0
      for k in range(window):
        if seq1[i+k] == seq2[j+k]:
          match_count += 1
      if match_count >= threshold:
        matrix[i][j] = 1
  return matrix



def compare_algorithms(seq1, seq2, window, threshold):
  """Compares the naive and optimized dot matrix algorithms."""

  # Naive approach
  start_time = time.time()
  matrix_naive = dotMatrix(seq1, seq2, window, threshold)
  end_time = time.time()
  naive_time = end_time - start_time

  # Optimized approach
  start_time = time.time()
  matrix_optimized = optimized_dot_matrix(seq1, seq2, window, threshold)
  end_time = time.time()
  optimized_time = end_time - start_time

  return naive_time, optimized_time

# Example usage
seq1 = "AGCTTAGCTAGGCTAATCGGATCGGCTTAGCTAAGCTTAGGCT"  # Replace with your protein sequence 1
seq2 = "AGCTTAGCTAGGCTAATCGGATCGGCTTAGCTAAGCTTAGGCT"  # Replace with your protein sequence 2
window_size = 10
threshold = 8

naive_time, optimized_time = compare_algorithms(seq1, seq2, window_size, threshold)

print(f"Naive algorithm time: {naive_time:.6f} seconds")
print(f"Optimized algorithm time: {optimized_time:.6f} seconds")


# Generate the dot matrix using the optimized approach
matrix = optimized_dot_matrix(seq1, seq2, window_size, threshold)

plt.figure(figsize=(10, 10))
plt.imshow(matrix, cmap='binary', interpolation='nearest')
plt.title('Dot Matrix Comparison (Optimized)')
plt.xlabel('Sequence 2 Position')
plt.ylabel('Sequence 1 Position')
plt.show()


# *****************************************************************************************************************************************************************************************************************************
# prompt: •	For each pairwise comparison, identify and extract regions of significant similarity. Define a conserved region as a region that is significantly matched in all pairwise comparisons.
# •	Create a combined dot matrix visualization that highlights regions conserved across all sequences. Overlay this with annotations to show the conserved regions.
# •	Plot individual dot matrices for each pairwise comp

import matplotlib.pyplot as plt
import numpy as np
def find_conserved_regions(matrices, seq_len, window, threshold):
  """Identifies conserved regions across multiple pairwise comparisons."""
  conserved_regions = []
  for i in range(seq_len - window + 1):
    conserved_at_i = True
    for matrix in matrices:
      conserved_in_matrix = False
      for j in range(seq_len - window + 1):
        if matrix[i][j] == 1:
          conserved_in_matrix = True
          break
      if not conserved_in_matrix:
        conserved_at_i = False
        break

    if conserved_at_i:
      conserved_regions.append(i)

  return conserved_regions


# Assuming you have a list of sequences and have generated pairwise dot matrices:
# For example, you can have a list of matrices: matrices = [matrix_tata_icici, matrix_tata_suzlon, matrix_icici_suzlon]

# Assuming seq1, seq2 are your sequences
seq_len = len(seq1)

# Generate pairwise matrices
matrix_tata_icici = optimized_dot_matrix(seq1, seq2, windowSize, threshold)
matrices = [matrix_tata_icici]  # Add more matrices as needed


# Find conserved regions
conserved_regions = find_conserved_regions(matrices, seq_len, windowSize, threshold)

# Create a combined dot matrix with conserved region highlights
combined_matrix = np.zeros((seq_len - windowSize + 1, seq_len - windowSize + 1))
for i in range(seq_len - windowSize + 1):
  for j in range(seq_len - windowSize + 1):
    if any(matrix[i][j] == 1 for matrix in matrices):
      combined_matrix[i][j] = 1

plt.figure(figsize=(10, 10))
plt.imshow(combined_matrix, cmap='binary', interpolation='nearest')
plt.title('Combined Dot Matrix with Conserved Regions')
plt.xlabel('Sequence Position')
plt.ylabel('Sequence Position')

# Annotate conserved regions
for region_start in conserved_regions:
  plt.axvline(x=region_start, color='r', linestyle='--', linewidth=1)
  plt.axhline(y=region_start, color='r', linestyle='--', linewidth=1)

plt.show()


# Plot individual dot matrices
for matrix, title in zip(matrices, ["Tata vs ICICI"]):
  plt.figure(figsize=(10, 10))
  plt.imshow(matrix, cmap='binary', interpolation='nearest')
  plt.title(title)
  plt.xlabel('Sequence Position')
  plt.ylabel('Sequence Position')
  plt.show()
