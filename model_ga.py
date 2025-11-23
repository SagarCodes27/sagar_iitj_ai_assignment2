import torch, random, os, json
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

from model_cnn import CNN

# Define the search space for CNN architecture
class CNNSearchSpace:
    def __init__(self):
        self.conv_layers = [1, 2, 3, 4]
        self.filters = [16, 32, 64, 128]
        self.kernel_sizes = [3, 5, 7]
        self.pool_types = ['max', 'avg']
        self.activations = ['relu', 'leaky_relu']
        self.fc_units = [64, 128, 256, 512]

# Encode architecture as a chromosome (gene representation)
class Architecture:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes
        self.fitness = 0
        self.accuracy = 0
        self.best_epoch = 0
    
    def random_genes(self):
        space = CNNSearchSpace()
        num_conv = random.choice(space.conv_layers)
        
        genes = {
            'num_conv': num_conv,
            'conv_configs': [],
            'pool_type': random.choice(space.pool_types),
            'activation': random.choice(space.activations),
            'fc_units': random.choice(space.fc_units)
        }
        
        for _ in range(num_conv):
            genes['conv_configs'].append({
                'filters': random.choice(space.filters),
                'kernel_size': random.choice(space.kernel_sizes)
            })
        
        return genes
    
    def __repr__(self):
        return f"Arch(conv={self.genes['num_conv']}, acc={self.accuracy:.4f})"

# Genetic Algorithm Operations
class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_architecture = None
        self.search_space = CNNSearchSpace()
    
    def initialize_population(self):
        self.population = [Architecture() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, architecture, train_loader, val_loader, device, epochs=50):
        """Train and evaluate a single architecture with GPU optimization"""
        try:
            model = CNN(architecture.genes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            # Training with early stopping
            best_acc = 0
            patience = 15  # More patience for full dataset
            patience_counter = 0
            best_epoch = 1

            for epoch in range(1, epochs+1):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                scheduler.step()

                # Evaluation
                model.eval()
                correct = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / len(val_loader.dataset)
                if accuracy > best_acc:
                    patience_counter = 0
                    best_acc = accuracy
                    best_epoch = epoch
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

            # Q2B: Separate parameter counting for Conv and FC layers
            conv_params = 0
            fc_params = 0

            for name, param in model.named_parameters():
                if 'features' in name:
                    conv_params += param.numel()
                elif 'classifier' in name:
                    fc_params += param.numel()

            # Normalize by 1 million for penalty calculation
            conv_penalty = conv_params / 1e6
            fc_penalty = fc_params / 1e6

            # Q2B: Weighted penalty (3:1 ratio - Conv is more computationally expensive)
            weight_conv = 0.015  # Higher weight for conv (more FLOPs per param)
            weight_fc = 0.005   # Lower weight for FC (fewer FLOPs per param)

            complexity_penalty = weight_conv * conv_penalty + weight_fc * fc_penalty

            # Log parameter details for Q2B verification
            print(f"    Conv params: {conv_params:,} | FC params: {fc_params:,}", flush=True)
            print(f"    Conv penalty: {conv_penalty:.6f} | FC penalty: {fc_penalty:.6f}", flush=True)
            print(f"    Weighted penalty: {complexity_penalty:.6f} | Best epoch: {best_epoch}", flush=True)

            # Cleanup GPU memory
            del model
            if 'inputs' in dir():
                del inputs, outputs, labels
            torch.cuda.empty_cache()

            architecture.accuracy = best_acc
            architecture.best_epoch = best_epoch
            architecture.fitness = best_acc - complexity_penalty

            return architecture.fitness

        except Exception as e:
            print(f"Error evaluating architecture: {e}", flush=True)
            import traceback
            traceback.print_exc()
            architecture.fitness = 0
            architecture.accuracy = 0
            return 0
    
    def selection(self):
        """Roulette-Wheel selection based on relative fitness"""
        selected = []

        fitness_values = [arch.fitness for arch in self.population]
        min_fitness = min(fitness_values)

        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        else:
            adjusted_fitness = [f + 1e-6 for f in fitness_values]

        total_fitness = sum(adjusted_fitness)

        relative_fitness = [f / total_fitness for f in adjusted_fitness]

        # Log Roulette-Wheel selection details
        print(f"\n  Roulette-Wheel Selection Details:", flush=True)
        print(f"  Fitness values: {[f'{f:.4f}' for f in fitness_values]}", flush=True)
        print(f"  Min fitness: {min_fitness:.4f}, Total fitness: {total_fitness:.4f}", flush=True)
        print(f"  Relative fitness (selection probabilities): {[f'{rf:.4f}' for rf in relative_fitness]}", flush=True)

        cumulative_probabilities = []
        cumulative_sum = 0
        for prob in relative_fitness:
            cumulative_sum += prob
            cumulative_probabilities.append(cumulative_sum)

        for _ in range(self.population_size):
            r = random.random()
            for i, cumulative_prob in enumerate(cumulative_probabilities):
                if r <= cumulative_prob:
                    selected.append(self.population[i])
                    break

        return selected
    
    def crossover(self, parent1, parent2):
        """Single-point crossover for architectures"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1_genes = deepcopy(parent1.genes)
        child2_genes = deepcopy(parent2.genes)
        
        # Crossover number of conv layers and pool type
        if random.random() < 0.5:
            child1_genes['num_conv'], child2_genes['num_conv'] = child2_genes['num_conv'], child1_genes['num_conv']
        
        # Crossover pool type and activation
        if random.random() < 0.5:
            child1_genes['pool_type'], child2_genes['pool_type'] = child2_genes['pool_type'], child1_genes['pool_type']
            child1_genes['activation'], child2_genes['activation'] = child2_genes['activation'], child1_genes['activation']
        
        # Adjust conv_configs to match num_conv
        min_len = min(child1_genes['num_conv'], len(child1_genes['conv_configs']))
        child1_genes['conv_configs'] = child1_genes['conv_configs'][:min_len]
        while len(child1_genes['conv_configs']) < child1_genes['num_conv']:
            child1_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        min_len = min(child2_genes['num_conv'], len(child2_genes['conv_configs']))
        child2_genes['conv_configs'] = child2_genes['conv_configs'][:min_len]
        while len(child2_genes['conv_configs']) < child2_genes['num_conv']:
            child2_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        return Architecture(child1_genes), Architecture(child2_genes)
    
    def mutation(self, architecture):
        """Mutate architecture genes"""
        if random.random() > self.mutation_rate:
            return architecture
        
        genes = deepcopy(architecture.genes)
        mutation_type = random.choice(['conv_param', 'num_layers', 'pool_activation', 'fc_units'])
        
        if mutation_type == 'conv_param' and genes['conv_configs']:
            # Mutate a random conv layer
            idx = random.randint(0, len(genes['conv_configs']) - 1)
            genes['conv_configs'][idx]['filters'] = random.choice(self.search_space.filters)
            genes['conv_configs'][idx]['kernel_size'] = random.choice(self.search_space.kernel_sizes)
        
        elif mutation_type == 'num_layers':
            # Change number of conv layers
            genes['num_conv'] = random.choice(self.search_space.conv_layers)
            # Adjust conv_configs
            if genes['num_conv'] > len(genes['conv_configs']):
                for _ in range(genes['num_conv'] - len(genes['conv_configs'])):
                    genes['conv_configs'].append({
                        'filters': random.choice(self.search_space.filters),
                        'kernel_size': random.choice(self.search_space.kernel_sizes)
                    })
            else:
                genes['conv_configs'] = genes['conv_configs'][:genes['num_conv']]
        
        elif mutation_type == 'pool_activation':
            genes['pool_type'] = random.choice(self.search_space.pool_types)
            genes['activation'] = random.choice(self.search_space.activations)
        
        elif mutation_type == 'fc_units':
            genes['fc_units'] = random.choice(self.search_space.fc_units)
        
        return Architecture(genes)
    
    def evolve(self, train_loader, val_loader, device, run=1):
        """Main evolutionary loop"""
        parent = os.path.abspath('')
        run_dir = os.path.join(parent, 'outputs', f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)

        self.initialize_population()
        print(f"Starting with {self.population_size} Population:\n{self.population}\n", flush=True)
        
        for generation in range(self.generations):
            print(f"\n{'='*60}", flush=True)
            print(f"Generation {generation + 1}/{self.generations}", flush=True)
            print(f"{'='*60}", flush=True)
            
            # Evaluate fitness
            for i, arch in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{self.population_size}...", end=' ', flush=True)
                fitness = self.evaluate_fitness(arch, train_loader, val_loader, device)
                print(f"Fitness: {fitness:.4f}, Accuracy: {arch.accuracy:.4f}", flush=True)
            
            # Sort by fitness score
            print(f"\nSorting population in terms of fitness score (high -> low) ...", flush=True)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if self.best_architecture is None or self.population[0].fitness > self.best_architecture.fitness:
                self.best_architecture = deepcopy(self.population[0])
            
            print(f"Best in generation: {self.population[0]}\n", flush=True)
            print(f"Best overall: {self.best_architecture}", flush=True)
            
            # Selection
            print(f"\nPerforming Roulette-Wheel selection of total population: {self.population_size} ...", flush=True)
            selected = self.selection()
            
            # Crossover and Mutation
            print(f"Performing Crossover & Mutation ...", flush=True)
            next_generation = []
            
            # Elitism: keep top 2 architectures
            print(f"Elitism: Keeping top 2 architectures in next generation.", flush=True)
            next_generation.extend([deepcopy(self.population[0]), deepcopy(self.population[1])])
            
            while len(next_generation) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            
            self.population = next_generation
            print(f"Next Generation: {self.population}", flush=True)
            with open(os.path.join(run_dir, f"generation_{generation}.jsonl"), 'w') as f:
                for obj in self.population:
                    f.write(json.dumps(obj.genes) + '\n')
        
        return self.best_architecture