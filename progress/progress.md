# AI/ML Engineering Journey — Progress

## Current Status

- **Phase:** Phase 0 — Foundations
- **Current Day:** Day 2
- **Status:** On Track
- **Primary objective:** Master core Python syntax, collections, and control flow through hands-on exercises and a capstone project.

## Day 1 — Environment and Setup

### Planned

1. Establish a working Python environment.
2. Use a project-local virtual environment.
3. Install the core packages required for the early AI/ML journey.
4. Verify Python, pip, scientific/ML imports, and JupyterLab.
5. Establish the initial repository structure and `.gitignore`.
6. Record evidence of the completed setup.
7. Finalize Day 1 documentation before moving to Day 2.

### Completed

- Python 3.13.16 installed and selected for the project environment.
- A fresh project-local `.venv` was created.
- The previous `.venv313` environment was replaced with the standard `.venv`.
- The environment was verified to use the intended Python interpreter.
- Core packages installed successfully:
  - NumPy
  - Pandas
  - SciPy
  - scikit-learn
  - Matplotlib
  - Seaborn
  - JupyterLab and its dependencies
- Initial repository structure created.
- `.gitignore` established with Python/virtual-environment and development exclusions.

### Environment Issue Resolved

The original Python 3.14 environment encountered native SciPy compatibility/build issues on Windows. The environment was rebuilt using Python 3.13.16, after which the required scientific/ML packages could be installed in a clean project-local virtual environment.

### Evidence

The final environment validation should be recorded in:

`00-foundations/python/day-01-environment-validation.ipynb`

The notebook validates:

- Python interpreter/version.
- pip availability.
- NumPy import and basic operation.
- Pandas import and basic DataFrame operation.
- SciPy import.
- scikit-learn import.
- Matplotlib import.
- Seaborn import.
- Jupyter installation/version availability.

### Documentation State

- Root `README.md`: established.
- `requirements.txt`: established with the core Day 1 environment dependencies.
- Day 1 validation notebook: established.
- `progress/progress.md`: updated with Day 1 status and evidence.
- Separate Day 1 Markdown report: intentionally not created; the handoff recommends keeping daily work in the relevant learning/project folder and using `progress/progress.md` for progress tracking.

### Understanding Check

Day 1 is an environment/setup milestone rather than a Python-learning milestone.

The important outcome is that the development environment can reliably support the upcoming Python, NumPy, Pandas, statistics, and machine-learning work.

## Day 2 — Python Basics & CLI Expense Calculator

### Planned

1. Cover core Python syntax: variables, data types (`int`, `float`), and string formatting.
2. Master collections: lists `[]`, dictionaries `{}`, tuples `()`, and sets `{}`.
3. Implement control flow: conditionals (`if`/`elif`/`else`) and loops (`for`/`while`).
4. Build a capstone CLI Expense Calculator.

### Completed

- Practiced variable assignment, numeric operations, and f-string formatting.
- Structured complex records using dictionaries nested within lists.
- Used sets to isolate unique elements and tuples for immutable values.
- Built `for` loops to iterate over collections and `while` loops to drive an interactive terminal menu.
- Built and verified a CLI Expense Calculator enabling users to add expenses, display logged entries, and compute dynamic total spending.

### Evidence

All concepts and code implementations are recorded in:

`00-foundations/python/day-02-python-basics.ipynb`

### Repository Updates

```text
00-foundations/python/day-02-python-basics.ipynb
progress/progress.md
README.md
```

### Suggested Commit

```text
feat: complete day 02 python basics and cli expense calculator
```

### Next

After the validation evidence is complete, proceed to Day 2.

Day 3 should involve Python functions
