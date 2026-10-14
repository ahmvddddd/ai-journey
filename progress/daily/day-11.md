# Day 11 — Matrix Operations & Linear Algebra Basics

## Status

* **Completion:** Completed
* **Topic:** Matrix Multiplication, Linear System Solving, Eigenvalues, and Least Squares
* **Time spent:** 2 hours

## Objectives

* [x] Execute matrix multiplication using the `@` operator and `np.dot()` with proper shape dimension matching
* [x] Understand linear transformations, matrix transpose (`A.T`), and determinants (`np.linalg.det`)
* [x] Solve square linear systems $A \cdot \vec{x} = \vec{b}$ directly using `np.linalg.solve()`
* [x] Compute eigenvalues and eigenvectors using `np.linalg.eig()`
* [x] Approximate solutions for overdetermined non-square systems using `np.linalg.lstsq()`

## Work Completed

Successfully explored and applied core linear algebra operations in NumPy:
* **Matrix Multiplication:** Verified outer and inner dimension rules for matrix multiplication ($(m \times n) @ (n \times p) \rightarrow (m \times p)$) using `@`.
* **Linear System Solving:** Applied `np.linalg.solve()` to solve $A \vec{x} = \vec{b}$ efficiently without calculating explicit matrix inverses.
* **Eigendecomposition:** Calculated eigenvalues and eigenvectors for linear transformations using `np.linalg.eig()`.
* **Least-Squares Approximation:** Utilized `np.linalg.lstsq()` to handle overdetermined system calculations where matrices are non-square.

## Evidence

* Interactive numerical computing exercises executed successfully in the workspace.

## Lessons Learned

* **Direct Solver Efficiency:** `np.linalg.solve()` is numerically stable and computationally faster than multiplying by an explicit inverse matrix (`np.linalg.inv()`).
* **Square Matrix Requirement:** `np.linalg.inv()` strictly requires square matrices; non-square or overdetermined systems must use least-squares solvers like `np.linalg.lstsq()`.

## Next Session

**Day 12:** NumPy Mini-Project.

## Suggested Commit

`feat: complete day 11 matrix operations and linear algebra with np.linalg`