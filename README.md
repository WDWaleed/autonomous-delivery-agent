# CS-370 AI Open Ended Lab: Autonomous Delivery Agent

Autonomous delivery agent operating in a grid city under terrain costs, stochastic movement (slip dynamics), dynamic city twists, and adversarial competition.

## 1. Environment Setup

### Prerequisites
- Python $\ge 3.11$ (Python 3.12 recommended)
- `pip`

### Virtual Environment Setup

#### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

#### Linux / macOS
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## 2. Testing

Run the test suite:
```bash
pytest -q
```
To run the full suite including slow tests:
```bash
pytest -m slow
```

## 3. Reproduction Pipeline

Reproduce all experiments, tables, and figures:
```bash
# Quick mode for fast validation
python run_all.py --quick

# Full experimental reproduction
python run_all.py
```

## 4. Repository Layout
- `delivery/`: Core agent, world, search, game, and LLM packages.
- `instances/`: Canonical problem instance generated from group CMS IDs.
- `tests/`: Test suite ensuring algorithmic correctness, invariants, and rules compliance.
- `docs/`: Technical specifications, decision logs, and proof documentation.
- `results/`: Stamped figures, tables, logs, and experiment numbers.
- `report/`: IEEE format 8–10 page report source and figures.
