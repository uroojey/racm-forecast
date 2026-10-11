"""Causal graph learner module using NOTEARS continuous optimization."""

from typing import List, Optional, Dict, Any
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import networkx as nx


class CausalLearner:
    """Discovers causal DAGs from observational time-series data using NOTEARS.
    
    NOTEARS formulates DAG learning as a continuous optimization problem:
        min_{W} 0.5 / n * ||X - X W||_F^2 + lambda1 * ||W||_1
        s.t.    h(W) = Tr(exp(W * W)) - d = 0
    where W is a weighted adjacency matrix and h(W) = 0 characterizes acyclicity.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _loss_and_grad(
        w_vec: np.ndarray,
        X: np.ndarray,
        d: int,
        n: int,
    ) -> tuple[float, np.ndarray]:
        """Calculates least squares loss and gradient with respect to W.
        
        Args:
            w_vec: Flattened adjacency matrix of shape (d * d,).
            X: Data matrix of shape (n, d).
            d: Number of variables/nodes.
            n: Number of samples.

        Returns:
            loss: Mean squared error 0.5 / n * ||X - XW||_F^2.
            grad: Flattened gradient of shape (d * d,).
        """
        W = w_vec.reshape((d, d))
        # R = X - X W
        R = X - X @ W
        loss = 0.5 / n * np.sum(R ** 2)
        # grad = -1/n * X^T (X - X W) = -1/n * X^T R
        grad = - (1.0 / n) * (X.T @ R)
        return loss, grad.flatten()

    @staticmethod
    def _h_and_grad(w_vec: np.ndarray, d: int) -> tuple[float, np.ndarray]:
        """Calculates acyclicity constraint h(W) and its gradient.
        
        h(W) = Tr(expm(W * W)) - d
        grad_W h(W) = 2 * (expm(W * W))^T * W
        """
        W = w_vec.reshape((d, d))
        M = W * W  # Hadamard product (elementwise)
        E = expm(M)
        h = np.trace(E) - d
        # Gradient with respect to W:
        # d(Tr(expm(W * W)))/dW_ij = (E^T)_ij * 2 * W_ij = 2 * E.T * W
        grad = 2.0 * (E.T * W)
        return h, grad.flatten()

    def fit(
        self,
        X: np.ndarray,
        lambda1: float = 0.01,
        max_iter: int = 100,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """Fits NOTEARS on observational data matrix X.
        
        Args:
            X: Observational temporal data matrix of shape (n_samples, n_variables).
            lambda1: L1 regularization parameter.
            max_iter: Maximum number of outer iterations for Augmented Lagrangian.
            threshold: Edge weight threshold to eliminate weak edges and ensure a clean DAG.

        Returns:
            W: Thresholded causal adjacency matrix of shape (n_variables, n_variables).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        else:
            X = X.astype(np.float64)

        n, d = X.shape
        w_est = np.zeros(d * d, dtype=np.float64)
        rho, alpha, h = 1.0, 0.0, np.inf
        rho_max = 1e16

        # Bounds: set diagonal elements of W to 0 (no self-loops)
        # Using bounds for L-BFGS-B: (0, 0) for diagonal entries, (-inf, inf) elsewhere
        bounds = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    bounds.append((0.0, 0.0))
                else:
                    bounds.append((None, None))

        for _ in range(max_iter):
            w_new, h_new = None, None

            def objective(w_vec: np.ndarray) -> tuple[float, np.ndarray]:
                # Loss & grad
                loss, grad_loss = self._loss_and_grad(w_vec, X, d, n)
                # h & grad_h
                h_val, grad_h = self._h_and_grad(w_vec, d)
                
                # Augmented Lagrangian:
                # obj = loss + lambda1 * ||W||_1 + alpha * h + 0.5 * rho * h^2
                # In smooth optimization, L1 penalty can be approximated or handled via subgradient/sign
                # Here we use standard L1 subgradient or smooth approximation
                # Standard NOTEARS uses variable splitting (W = W+ - W-) or subgradient.
                # For L-BFGS-B with smooth obj:
                l1_penalty = lambda1 * np.sum(np.abs(w_vec))
                l1_grad = lambda1 * np.sign(w_vec)
                
                obj = loss + l1_penalty + alpha * h_val + 0.5 * rho * (h_val ** 2)
                grad = grad_loss + l1_grad + (alpha + rho * h_val) * grad_h
                return obj, grad

            res = minimize(
                objective,
                w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 150},
            )
            w_new = res.x
            h_new, _ = self._h_and_grad(w_new, d)

            if h_new > 0.25 * h:
                rho *= 10.0
            else:
                break
            if rho >= rho_max:
                break

            w_est = w_new
            alpha += rho * h_new
            h = h_new

        w_est = res.x if w_new is None else w_new
        W = w_est.reshape((d, d))
        
        # Zero-out self-loops
        np.fill_diagonal(W, 0.0)

        # Thresholding small weights
        W[np.abs(W) < threshold] = 0.0

        return W

    @staticmethod
    def is_dag(W: np.ndarray) -> bool:
        """Checks if adjacency matrix W represents a directed acyclic graph.
        
        Args:
            W: Adjacency matrix of shape (d, d).

        Returns:
            True if graph is a DAG, False otherwise.
        """
        if not isinstance(W, np.ndarray):
            W = np.asarray(W)
        G = nx.DiGraph(W != 0)
        return nx.is_directed_acyclic_graph(G)

    @staticmethod
    def get_causal_edges(
        W: np.ndarray,
        node_names: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Extracts directed causal edges with weights above threshold.
        
        Args:
            W: Adjacency matrix of shape (d, d) where W[i, j] indicates edge from i to j.
            node_names: Optional list of variable names corresponding to indices.
            threshold: Minimum absolute weight threshold.

        Returns:
            List of dicts: [{"source": u, "target": v, "weight": float(W[i, j])}]
        """
        d = W.shape[0]
        if node_names is None:
            node_names = [f"X{i}" for i in range(d)]

        edges = []
        for i in range(d):
            for j in range(d):
                if i != j and abs(W[i, j]) >= threshold:
                    edges.append({
                        "source": node_names[i],
                        "target": node_names[j],
                        "weight": float(W[i, j]),
                    })
        return edges

    @staticmethod
    def merge_with_knowledge_graph(
        W: np.ndarray,
        kg: nx.MultiDiGraph,
        node_names: List[str],
        threshold: float = 0.1,
    ) -> nx.MultiDiGraph:
        """Adds discovered causal edges into a copy of the knowledge graph.
        
        Args:
            W: Adjacency matrix of shape (d, d).
            kg: Existing NetworkX MultiDiGraph.
            node_names: List of variable names matching rows/cols of W.
            threshold: Minimum weight threshold for causal edges.

        Returns:
            enriched_kg: A new MultiDiGraph with added causal edges.
        """
        enriched_kg = kg.copy()
        d = W.shape[0]
        for i in range(d):
            for j in range(d):
                if i != j and abs(W[i, j]) >= threshold:
                    u = node_names[i]
                    v = node_names[j]
                    enriched_kg.add_edge(
                        u,
                        v,
                        edge_type="causes",
                        weight=float(W[i, j]),
                    )
        return enriched_kg
