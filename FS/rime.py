import numpy as np
from numpy.random import rand
from FS.functionHO import Fun

def initiallization(N, dim, ub, lb):
    # Random initialization of RIME population
    return lb + (ub - lb) * np.random.rand(N, dim)

def normr(mat):
    # Normalization by row
    row_means = np.mean(mat, axis=1, keepdims=True)
    row_stds = np.std(mat, axis=1, keepdims=True)
    return (mat - row_means) / row_stds if row_stds.any() != 0 else mat

def jfs(xtrain, ytrain, opts):
    
    """
    RIME algorithm for joint feature selection.
    :param N: Number of agents
    :param Max_iter: Maximum number of iterations
    :param lb, ub: Lower and upper bounds for each dimension
    :param dim: Number of features (dimensions)
    :param fobj: Fitness function (e.g., Fun from gwo.py)
    :return: Best feature subset, convergence curve
    """
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    Max_iter = max_iter
    # Dimension
    dim = np.size(xtrain, 1)
    
    # Define fobj as Fun from gwo.py
    def fobj(features,opts=opts,xtrain=xtrain,ytrain=ytrain):
        return Fun(xtrain, ytrain, features, opts)
    # Initialize parameters
    Best_rime = np.zeros(dim)
    Best_rime_rate = float('inf')  # For minimization problems
    Rimepop = initiallization(N, dim, ub, lb)
    Lb = lb * np.ones(dim)
    Ub = ub * np.ones(dim)
    it = 1
    Convergence_curve = np.zeros(Max_iter)
    Rime_rates = np.zeros(N)
    newRime_rates = np.zeros(N)
    W = 5  # Soft-RIME parameter
    
    # Evaluate initial population
    for i in range(N):
        Rime_rates[i] = fobj(Rimepop[i, :])
        if Rime_rates[i] < Best_rime_rate:
            Best_rime_rate = Rime_rates[i]
            Best_rime = Rimepop[i, :]
    
    # Main loop
    while it <= Max_iter:
        # Update RimeFactor and E
        RimeFactor = ((np.random.rand() - 0.5) * 2 * 
                     np.cos(np.pi * it / (Max_iter / 10)) * 
                     (1 - np.round(it * W / Max_iter) / W))
        E = (it / Max_iter) ** 0.5
        newRimepop = Rimepop.copy()
        normalized_rime_rates = normr(Rime_rates.reshape(-1, 1))
        
        # Update newRimepop based on strategies
        for i in range(N):
            for j in range(dim):
                # Soft-RIME search
                r1 = np.random.rand()
                if r1 < E:
                    newRimepop[i, j] = Best_rime[j] + RimeFactor * ( (Ub[j] - Lb[j]) * np.random.rand() + Lb[j] )
                
                # Hard-RIME puncture
                r2 = np.random.rand()
                if r2 < normalized_rime_rates[i]:
                    newRimepop[i, j] = Best_rime[j]
        
        # Boundary absorption and evaluation
        for i in range(N):
            # Apply boundary conditions
            newRimepop[i] = np.clip(newRimepop[i], Lb, Ub)
            # Evaluate new fitness
            newRime_rates[i] = fobj(newRimepop[i, :])
            # Update if better
            if newRime_rates[i] < Rime_rates[i]:
                Rime_rates[i] = newRime_rates[i]
                Rimepop[i] = newRimepop[i]
                # Update best
                if newRime_rates[i] < Best_rime_rate:
                    Best_rime_rate = Rime_rates[i]
                    Best_rime = Rimepop[i]
        
        # Record best fitness
        Convergence_curve[it - 1] = Best_rime_rate
        it += 1
    
    # Extract selected features
    Gbin = np.round(Best_rime)  # Binary conversion
    sel_index = np.where(Gbin == 1)[0]
    num_feat = len(sel_index)
    
    # Create dictionary for results
    rime_data = {
        'sf': sel_index,
        'c': Convergence_curve.reshape(1,-1),
        'nf': num_feat
    }
    
    return rime_data

# Example usage with Fun (from gwo.py)
def main():
    # Example parameters (integrate with your own datasets and options)
    opts = {
        'N': 20,  # Number of agents
        'T': 100  # Maximum iterations
    }
    lb = 0
    ub = 1
    dim = 10  # Number of features
    N = opts['N']
    Max_iter = opts['T']
    xtrain = np.random.rand(100, dim)  # Example training data
    ytrain = np.random.randint(0, 2, 100)  # Example labels
    
    
    
    rime_results = RIME(N, Max_iter, lb, ub, dim, fobj)
    print("Selected features:", rime_results['sf'])
    print("Number of features:", rime_results['nf'])
    print("Convergence curve:", rime_results['c'])

if __name__ == '__main__':
    main()