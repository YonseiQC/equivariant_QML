import numpy as np
import itertools
from itertools import permutations
import math

def create_permutation_matrix(cycle, num_qubit):
    n = 2**num_qubit
    P = np.zeros((n, n))
    
    perm = list(range(num_qubit))
    if len(cycle) > 1:
        for i in range(len(cycle)):
            perm[cycle[i]] = cycle[(i+1) % len(cycle)]
    
    for i in range(n):
        binary = [(i >> bit) & 1 for bit in range(num_qubit)]
        
        new_binary = [0] * num_qubit
        for j in range(num_qubit):
            new_binary[perm[j]] = binary[j]
        
        new_i = sum(bit * (2**pos) for pos, bit in enumerate(new_binary))
        P[new_i, i] = 1
    
    return P

def expand_cross_permutation(pairs, is_plus):
    terms = []
    signs = []
    
    choices = []
    for pair in pairs:
        choices.append([pair[0], pair[1]])
    
    for combination in itertools.product(*choices):
        if is_plus:
            sign = 1
        else:
            wire_sum = sum(combination)
            sign = 1 if wire_sum % 2 == 0 else -1
        
        terms.append(combination)
        signs.append(sign)
    
    return terms, signs

def generate_permutation_matrices(num_qubit):
    
    num_pairs = num_qubit // 2
    pairs = [(2*i, 2*i+1) for i in range(num_pairs)]
    
    max_perm = num_pairs
    
    for num_perm in range(2, max_perm + 1):
        
        pair_perms = list(permutations(pairs, num_perm))
        
        # Normalize by num_perm!
        normalization_factor = math.factorial(num_perm)
        
        print(f"num_perm = {num_perm}")
        print(f"Normalization factor (num_perm!): {normalization_factor}")
        
        plus_matrix = np.zeros((2**num_qubit, 2**num_qubit))
        
        for pair_perm in pair_perms:
            terms, signs = expand_cross_permutation(pair_perm, is_plus=True)
            
            for term, sign in zip(terms, signs):
                perm_matrix = create_permutation_matrix(term, num_qubit)
                plus_matrix += sign * perm_matrix
        
        # Normalize by num_perm!
        plus_matrix = plus_matrix / normalization_factor
        
        minus_matrix = np.zeros((2**num_qubit, 2**num_qubit))
        
        for pair_perm in pair_perms:
            terms, signs = expand_cross_permutation(pair_perm, is_plus=False)
            
            for term, sign in zip(terms, signs):
                perm_matrix = create_permutation_matrix(term, num_qubit)
                minus_matrix += sign * perm_matrix
        
        # Normalize by num_perm!
        minus_matrix = minus_matrix / normalization_factor

        np.save(f'perm_matrix_{num_qubit}_{num_perm}_plus_normalized.npy', plus_matrix)
        np.save(f'perm_matrix_{num_qubit}_{num_perm}_minus_normalized.npy', minus_matrix)
        
        print(f"Saved perm_matrix_{num_qubit}_{num_perm}_plus_normalized.npy and perm_matrix_{num_qubit}_{num_perm}_minus_normalized.npy")
        print(f"Matrix shape: {plus_matrix.shape}")
        print(f"Matrix normalized by {normalization_factor}")
        print("-" * 40)

# Run the function
generate_permutation_matrices(12)