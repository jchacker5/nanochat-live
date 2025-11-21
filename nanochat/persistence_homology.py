"""
Persistence Homology Tracker for topological feature tracking.

Tracks topological features (holes, loops) across scales using persistent homology,
which is preserved by the Čech-de Rham isomorphism.

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
    Section 3.4.1: Topological Deep Learning.
"""

import torch
import torch.nn as nn
import numpy as np

# Try to import scipy, but make it optional
try:
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PersistenceHomologyTracker(nn.Module):
    """
    Track persistent homology (topological features across scales).
    
    Uses Čech complexes at multiple scales to compute persistence diagrams,
    which summarize topological invariants preserved by Čech-de Rham.
    
    Args:
        max_dim: Maximum homology dimension to compute (0=components, 1=loops)
    
    Note: This is a simplified implementation. For production use, consider
    using ripser library for more accurate persistence computation.
    """
    
    def __init__(self, max_dim=1):
        """
        Args:
            max_dim: Maximum homology dimension to compute (0=components, 1=loops)
        """
        super().__init__()
        self.max_dim = max_dim
    
    def _count_components(self, adjacency):
        """Count connected components using DFS."""
        n = len(adjacency)
        visited = np.zeros(n, dtype=bool)
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in range(n):
                if adjacency[node, neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        return components
    
    def compute_betti_number(self, x, epsilon=0.1):
        """
        Compute Betti number using Čech complex.
        
        Args:
            x: (B, T, n_embd) embeddings
            epsilon: Radius for Čech complex
        
        Returns:
            betti: (B,) Betti numbers
        """
        B, T, C = x.shape
        betti_numbers = []
        
        for b in range(B):
            # Build Čech complex: connect points within epsilon distance
            points = x[b].detach().cpu().numpy()  # (T, C)
            
            if T < 2:
                betti_numbers.append(0)
                continue
            
            # Compute pairwise distances
            if HAS_SCIPY:
                try:
                    distances = squareform(pdist(points))
                except:
                    # Fallback: manual computation
                    distances = np.zeros((T, T))
                    for i in range(T):
                        for j in range(i+1, T):
                            dist = np.linalg.norm(points[i] - points[j])
                            distances[i, j] = dist
                            distances[j, i] = dist
            else:
                # Manual computation without scipy
                distances = np.zeros((T, T))
                for i in range(T):
                    for j in range(i+1, T):
                        dist = np.linalg.norm(points[i] - points[j])
                        distances[i, j] = dist
                        distances[j, i] = dist
            
            # Build adjacency matrix (within epsilon)
            adjacency = (distances < epsilon).astype(float)
            
            # Compute Betti_1 = rank(H_1) = edges - vertices + components
            # Simplified: count cycles in graph
            n_edges = int(np.sum(adjacency) // 2)  # Undirected
            n_vertices = T
            n_components = self._count_components(adjacency)
            
            # Betti_1 = edges - vertices + components (Euler characteristic)
            betti_1 = n_edges - n_vertices + n_components
            
            betti_numbers.append(max(0, int(betti_1)))
        
        return torch.tensor(betti_numbers, device=x.device, dtype=torch.long)
    
    def persistence_loss(self, x_before, x_after, epsilon=0.1):
        """
        Compute loss based on Betti number stability.
        
        Penalizes changes in topological features (Betti numbers).
        
        Args:
            x_before: (B, T, n_embd) embeddings before transformation
            x_after: (B, T, n_embd) embeddings after transformation
            epsilon: Radius for Čech complex
        
        Returns:
            loss: scalar
        """
        betti_before = self.compute_betti_number(x_before, epsilon)
        betti_after = self.compute_betti_number(x_after, epsilon)
        
        # Penalize changes in Betti numbers
        loss = torch.mean((betti_after.float() - betti_before.float()).pow(2))
        
        return loss

