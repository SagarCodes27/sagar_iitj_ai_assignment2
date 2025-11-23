# Neural Architecture Search using Genetic Algorithm - Modified Version

## Assignment 2 - Q1 Implementation
### Student Name: Sagar Shrivastava
### Roll number: G25AIT1144

##

This repository contains the modified NAS-GA code with the following enhancements:

### Q1A: Roulette-Wheel Selection
Modified the `selection()` function in `model_ga.py` to implement Roulette-Wheel selection instead of tournament selection. The implementation:
- Calculates relative fitness scores for all chromosomes in the population
- Handles negative fitness values by shifting all fitnesses to positive range
- Assigns proportional probabilities based on relative fitness
- Uses cumulative probability distribution for selection
- Higher fitness chromosomes have higher probability of being selected

### Q2B: Enhanced Fitness Function
Modified the `evaluate_fitness()` function in `model_ga.py` to separately consider convolutional and fully-connected layer parameters:
- Separately counts parameters in convolutional blocks (features) and FC layers (classifier)
- Applies different penalty weights: weight_conv = 0.015, weight_fc = 0.005
- Weight ratio (3:1) reflects computational cost difference between Conv and FC operations
- Conv operations are 3x more expensive due to spatial dimension processing and channel-wise operations

---

## Environment Setup

### Prerequisites
- **Python 3.10, 3.11, or 3.12** (Required for CUDA support)
- **NVIDIA GPU with CUDA 12.1** (Optional but recommended)

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

**For GPU (CUDA 12.1) - Recommended:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch: {torch.__version__}')"
```

Expected output for GPU setup:
```
CUDA available: True
PyTorch: 2.5.1+cu121
```

---

## Running the NAS-GA Algorithm

### Quick Start
```bash
# Activate venv first
.\venv\Scripts\Activate.ps1

# Run the algorithm
python nas_run.py
```

### Configuration
You can modify the following parameters in `nas_run.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 10 | Number of architectures per generation |
| `generations` | 5 | Number of evolutionary generations |
| `mutation_rate` | 0.3 | Probability of mutation |
| `crossover_rate` | 0.7 | Probability of crossover |

### Dataset
- Uses **partial CIFAR-10 dataset** (8,000 training + 1,500 validation images)
- Data augmentation: RandomCrop, RandomHorizontalFlip
- Automatic download on first run

---

## Output Files

All outputs are saved in `outputs/run_X/` directory:

| File | Description |
|------|-------------|
| `nas_run.log` | Detailed execution log with all required outputs |
| `generation_Y.jsonl` | Population genes for each generation |
| `best_arch.pkl` | Best architecture found (pickle format) |

### Log Contents (for submission)
The log file contains all required outputs:
- **Q1A**: Relative fitness scores and selection probabilities per generation
- **Q2B**: Conv params, FC params, penalties, and computed fitness per architecture

Example log output:
```
Evaluating architecture 1/20...
    Conv params: 216,368 | FC params: 264,970
    Conv penalty: 0.216368 | FC penalty: 0.264970
    Weighted penalty: 0.004570 | Best epoch: 25
Fitness: 0.7284, Accuracy: 0.7330

Roulette-Wheel Selection Details:
  Fitness values: ['0.7284', '0.6891', ...]
  Relative fitness (selection probabilities): ['0.1276', '0.1235', ...]
```

---

## Key Modifications

### File: model_ga.py

#### Selection Function (Lines 160-195)
Implements Roulette-Wheel selection with:
- Fitness normalization for handling negative values
- Cumulative probability calculation
- Probabilistic selection based on fitness proportions

#### Fitness Evaluation Function (Lines 67-158)
Enhanced fitness calculation with:
- Separate parameter counting for Conv and FC layers
- Learning rate scheduler (CosineAnnealingLR)
- GPU-optimized training loop
- Formula: `fitness = accuracy - (0.015 * conv_params/1e6 + 0.005 * fc_params/1e6)`

### File: nas_run.py
- Full CIFAR-10 dataset with data augmentation
- GPU-optimized DataLoaders (pin_memory, num_workers)
- Conditional settings for CPU fallback

---

## Troubleshooting

### CUDA not available
1. Check NVIDIA driver: `nvidia-smi`
2. Ensure Python version is 3.10-3.12 (not 3.13)
3. Reinstall PyTorch with CUDA: `pip uninstall torch torchvision && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Out of Memory
- Reduce `batch_size` in `nas_run.py`
- Reduce `population_size` in `nas_run.py`
- Reduce `subset` - `train & validate` in `nas_run.py`

---

## References
- Stack Overflow: Roulette Wheel Selection Algorithm
- PyTorch Documentation: https://pytorch.org/docs/stable/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
