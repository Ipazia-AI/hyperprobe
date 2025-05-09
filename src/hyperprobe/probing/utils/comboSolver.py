import torch
import itertools
import numpy as np

import heapq

class comboSolver():
    def __init__(self, domains, batch_size = None, device = None):
    
        # Set class variables
        self.domains = domains
        self.device = device

        # Get the codebook dimensions
        self.domain_dims = [len(domain) for domain in domains.values()]
        
        # Set the batch size
        self.batch_size = int(batch_size) if batch_size and batch_size > 0 else np.prod(self.domain_dims).item()
        
        # Generate all possible combinations
        print(f'[INFO] Initialize the compoSolver (domains = {len(domains.keys())}, batch_size = {self.batch_size}) by generating all possible combinations...')
        self.combinations_indices = torch.tensor(list(itertools.product(*(range(dim) for dim in self.domain_dims))))

        # Save the combinations names
        self.all_combinations_names, self.feature_combinations = self.generate_combinations()
        
        # Generate the unique objects via binding
        self.combinations, self.combinations_names = self.generate_unique_objects()
        print('[INFO] CompoSolver:', self.feature_combinations.shape[0], f'({self.feature_combinations.shape[0]:.0E}) unique combinations.')
        
        # Clear the cache
        torch.cuda.empty_cache()
        
    def generate_combinations(self):
        domains_values = [torch.from_numpy(domain.values) for domain in self.domains.values()]
        domain_labels = [domain.index.tolist() for domain in self.domains.values()]
        
        combination_names, combination_values = [], []
        for indices in self.combinations_indices:
            combination_names.append([domain_labels[i][index] for i, index in enumerate(indices)])
            combination_values.append(torch.stack([domains_values[i][index] for i, index in enumerate(indices)]))
        combination_values = torch.stack(combination_values)
        combination_names = np.array(combination_names)
        
        return combination_names, combination_values
    
    def generate_unique_objects(self):
        
        # Generate the unique objects via binding
        combinations, names = dict(), dict()
        for num_domains in range(1, self.feature_combinations.shape[1] + 1): 
            
            # Get features names
            unique_features, indices = np.unique(self.all_combinations_names[:, :num_domains], axis=0, return_index=True)
            names[num_domains] = unique_features[np.argsort(indices)].tolist()
        
            # Select only the relevant feature combinations
            bound_objects = torch.cat([
                torch.prod(combinations.to(self.device), dim=1, dtype = torch.int8)
                for combinations in torch.split(self.feature_combinations[np.sort(indices), :num_domains], self.batch_size)])
            combinations[num_domains] = bound_objects#.cpu()

        return combinations, names
    
    def _factorization(self, query, target):
        all_similarities = []
        for num_domains in range(self.feature_combinations.shape[1], 0, -1):
            
            # Find the target item within the combinations and remove it
            mask = torch.any(self.combinations[num_domains] != target, dim=1)
 
            # Compute the similarities
            similarities = torch.cat([
                torch.cosine_similarity(x1 = target, x2 = query * combinations)
                for combinations in torch.split(self.combinations[num_domains][mask].to(self.device), self.batch_size)
            ]).cpu()
            
            # Attach the similarities to the combination names, sort them and save the first 10
            all_similarities.extend([
                dict(features = names, sim = sim.item(), domains = num_domains)
                for names, sim in zip([name for i, name in enumerate(self.combinations_names[num_domains]) if i != torch.argwhere(~mask).item()], similarities)])
            
            # Check if we found a similarity equal to 1 (break the loop)
            if (similarities == 1).any():
                break
        
        # Sort all the similarities
        all_similarities = heapq.nlargest(20, all_similarities, 
                                        key=lambda x: (len(self.domain_dims) / x['domains']) * x['sim'])
                  
        # Find the best match
        best_match = all_similarities[0]
        
        return best_match, all_similarities
            
    def query(self, query, target):
        return self._factorization(query.to(self.device), target.to(self.device))
        