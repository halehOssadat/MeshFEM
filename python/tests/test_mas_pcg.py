"""Run with MeshFEM's python directory on PYTHONPATH."""
import unittest
import numpy as np
import scipy.sparse as sp
import MeshFEM
import sparse_matrices as sm
import block_sparse_hessian as bsh


def grid_system(n, mass, negative_node=-1):
    """Upper triangle of kron(L + mass I, S) for the 6-neighbor Laplacian L of an n^3 grid."""
    S = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]])
    path = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n))
    I = sp.identity(n)
    adj = sp.kron(sp.kron(path, I), I) + sp.kron(sp.kron(I, path), I) + sp.kron(sp.kron(I, I), path)
    deg = np.asarray(adj.sum(axis=1)).ravel() + mass
    if negative_node >= 0: deg[negative_node] *= -1
    K = sp.kron(sp.diags(deg) - adj, S)
    A = sm.SuiteSparseMatrix.fromSciPy(sp.triu(K).tocsc())
    A.symmetry_mode = sm.SymmetryMode.UPPER_TRIANGLE
    return A, K.tocsr()


class MASPCGTests(unittest.TestCase):
    def setUp(self):
        self.A, self.K = grid_system(6, 1e-2)
        self.n = self.K.shape[0]
        self.x_expected = np.sin(0.3 * np.arange(1, self.n + 1)) + 0.25

    def solve(self, A, mas, pins=(), sigma=0.0):
        pins = list(pins)
        x_expected = self.x_expected.copy()
        x_expected[pins] = 0
        b = self.K @ x_expected + sigma * x_expected
        b[pins] = 0

        mas.analyzePattern(A, pins)
        mas.compute(A, sigma)
        pcg = sm.PCG()
        pcg.relTol = 1e-11
        x = pcg.solve(sm.SymmetricMatrixOperator(A, pins, sigma), mas, b)
        stats = pcg.lastStats()
        self.assertTrue(stats.converged, stats)
        np.testing.assert_allclose(x, x_expected, atol=1e-8 * np.linalg.norm(x_expected))
        return stats

    def test_scalar_matrix(self):
        mas = sm.MASPreconditioner()
        mas.nodeBlockSize = 3
        for pins in [(), (0, 1, 2), (4, 100, 101)]:
            for sigma in [0.0, 0.5]:
                self.solve(self.A, mas, pins, sigma)
        self.assertEqual(mas.nodeBlockSizeUsed(), 3)
        self.assertGreaterEqual(mas.numLevels(), 2)

    def test_block_matrix(self):
        H = bsh.BlockCSCHessianBase.fromScalar(self.A, 3)
        mas = sm.MASPreconditioner()
        mas.partitionMethod = sm.MASPreconditioner.PartitionMethod.Greedy
        self.solve(H, mas, (4, 100, 101), 0.5)
        self.assertEqual(mas.nodeBlockSizeUsed(), 3)
        self.assertFalse(mas.usedMetis())

    def test_operator_and_preconditioner_apply(self):
        pins, sigma = [1, 5, 30], 0.25
        x = np.cos(0.7 * np.arange(self.n))
        x_free = x.copy()
        x_free[pins] = 0
        expected = self.K @ x_free + sigma * x_free
        expected[pins] = x[pins]
        op = sm.SymmetricMatrixOperator(self.A, pins, sigma)
        self.assertEqual(op.size(), self.n)
        np.testing.assert_allclose(op.apply(x), expected, rtol=1e-12)

        mas = sm.MASPreconditioner()
        mas.analyzePattern(self.A, pins)
        with self.assertRaises(RuntimeError): mas.apply(x)
        mas.compute(self.A, sigma)
        np.testing.assert_array_equal(mas.apply(x)[pins], x[pins])

    def test_indefinite(self):
        A, _ = grid_system(5, 1e-2, negative_node=17)
        mas = sm.MASPreconditioner()
        mas.nodeBlockSize = 3
        mas.analyzePattern(A)
        with self.assertRaises(RuntimeError): mas.compute(A)
        self.assertFalse(mas.isComputed())
        mas.compute(A, 100.0)
        self.assertTrue(mas.isComputed())


if __name__ == '__main__':
    unittest.main()
