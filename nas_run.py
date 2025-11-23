import torch, sys, os, pickle
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from model_ga import GeneticAlgorithm
from model_cnn import CNN

if __name__ == "__main__":
    parent = os.path.abspath('')
    os.makedirs(os.path.join(parent, 'outputs'), exist_ok=True)
    existing_runs = [d for d in os.listdir(os.path.join(parent, 'outputs')) if d.startswith('run_')]
    run_num = len(existing_runs) + 1
    os.makedirs(os.path.join(parent, 'outputs', f'run_{run_num}'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sys.stdout = open(os.path.join(parent, 'outputs', f'run_{run_num}', f'nas_run.log'), 'w')

    print(f"Using device: {device}", flush=True)

    # Load CIFAR-10 dataset (reduced for faster NAS)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Use only 10000 samples for quick NAS (~10 min on my 4070 Ti)
    train_subset = Subset(trainset, range(8000))
    val_subset = Subset(valset, range(1500))

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

    # Run NAS with GA
    ga = GeneticAlgorithm(
        population_size=10,
        generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7
    )

    best_arch = ga.evolve(train_loader, val_loader, device, run=run_num)

    print(f"\n{'='*60}", flush=True)
    print("FINAL BEST ARCHITECTURE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Genes: {best_arch.genes}", flush=True)
    print(f"Accuracy: {best_arch.accuracy:.4f}", flush=True)
    print(f"Fitness: {best_arch.fitness:.4f}", flush=True)

    # Build and test final model
    final_model = CNN(best_arch.genes).to(device)
    print(f"\nTotal parameters: {sum(p.numel() for p in final_model.parameters()):,}", flush=True)
    print(f"\nModel architecture:\n{final_model}", flush=True)

    with open(os.path.join(parent, 'outputs', f'run_{run_num}', f"best_arch.pkl"), 'wb') as f:
        pickle.dump(best_arch, f)

    sys.stdout = sys.__stdout__
    print(f"Done! Log saved to: outputs/run_{run_num}/nas_run.log")
