from os import path, listdir, sched_getaffinity
from collections import Counter, defaultdict
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from lightning import LightningDataModule
import numpy as np
import re
import torch
import json
import pickle
import zlib

from sklearn.model_selection import train_test_split
import torchhd

# LOCAL IMPORTS
from hyperprobe.encoder.utils.vsa_utils import create_vsa_encodings 

def load_jsonFile(file_path):
    with open(file_path, mode ='r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def load_embeddings(root_folder: str) -> list[dict]:
    
    
    # Load compressed file
    with open(path.join(root_folder, 'embeddings.pkl.zlib'), mode = "rb") as file:
       data = pickle.loads(zlib.decompress(file.read()))
       
    # Concepts 
    concept_path = path.join(root_folder, 'concepts.json')
    if path.exists(concept_path):
        with open(concept_path, mode = "r") as file:
            concepts = json.load(file)
            
        embeddings = [{'doc': doc, 'embeddings': emb, 'features': concepts[doc] if doc in concepts else []} for doc, emb in data.items()]
    else:
        embeddings = [{'doc': doc, 'embeddings': emb} for doc, emb in data.items()]
        
    print(f'\n[INFO] Loaded embeddings ({len(embeddings)}) from:', root_folder)
    
    return embeddings    

class llm2VSA_dataloader(LightningDataModule):
    def __init__(self, data, batch_size, split = 'random', val_size = 0.1, test_size = 0.1):
        super().__init__()
        
        # Load the data
        self.data = data
        
        # Hyperparameters
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.split = split
        
        # Constants
        self.random_seed = 101
        self.num_workers = torch.clip(torch.tensor(len(sched_getaffinity(0))), min = 0, max = 2)
        
    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
    
    def get_input_dim(self):
        return self.data.get_input_dim()
    
    def get_target_dim(self):
        return self.data.get_target_dim()
    
    def get_splitted_labels(self):
        subset_indices = {'train': self.train_set.indices, 'val': self.val_set.indices, 'test': self.test_set.indices}
        data_stats = {subset: self.data.texts[indices].tolist() for subset, indices in subset_indices.items()}
        return data_stats
        
    def setup(self, stage:str):
        
        # Split the data into train, test, and validation sets
        if self.split == 'random':
            self._random_split()
            print('\n[INFO] [Random] Splitting the data into train, validation, and test sets')
        elif self.split == 'cluster':
            self._cluster_split()
            print('\n[INFO] [Cluster-based] Splitting the data into train, validation, and test sets')
        elif self.split == 'predefined':
            self._preComputed_splits()
            print('\n[INFO] [Pre-computed based] Splitting the data into train, validation, and test sets')
        elif self.split == 'hard':
            self._portion_split()
            print('\n[INFO] [Portion-based] Splitting the data into train, validation, and test sets')
        else:
            raise ValueError(f"Invalid split type: {self.split}")
            
        print('\nINPUTS:', len(self.data), '--> TRAIN:', round(((len(self.train_set) / len(self.data)) * 100), 1), '%',
              '|| VALIDATION:', round(((len(self.val_set) / len(self.data)) * 100), 1), '%',
              '|| TEST:', round(((len(self.test_set) / len(self.data)) * 100), 1), '%', '\n')
        
    def _random_split(self):
        self.train_set, self.val_set, self.test_set = random_split(
            dataset = self.data, 
            generator = torch.Generator().manual_seed(self.random_seed),
            lengths = [1 - self.val_size - self.test_size, self.val_size, self.test_size])
        
    def _cluster_split(self):
        
        # Get the classes
        classes = np.array(self.data.domains)
        
        try: 
            # Get the indices for the train, validation, and test sets (two-step split)
            train_ids, test_ids = train_test_split(range(len(self.data)), 
                train_size = 1 - self.val_size - self.test_size,
                stratify = classes,
                shuffle = True, 
                random_state = self.random_seed)    
                        
            val_ids, test_ids = train_test_split(test_ids, 
                test_size = 0.5, 
                stratify = classes[test_ids], 
                shuffle = True, 
                random_state = self.random_seed)
            
             # Create the subsets
            self.train_set = Subset(self.data, train_ids)
            self.val_set = Subset(self.data, val_ids)
            self.test_set = Subset(self.data, test_ids)
            
        except ValueError:
            counter = np.unique(classes, return_counts = True)
            counter = dict(zip(counter[0].tolist(), counter[1].tolist()))
            print('\n[WARNING] The stratified split failed. Using the random split instead.\n', counter)
            
            self._random_split()   
            
    def _preComputed_splits(self):
        
        # Get the indices for the train, validation, and test sets
        train_ids = [idx for idx, item in enumerate(self.data.splits) if item == 'train']
        val_ids = [idx for idx, item in enumerate(self.data.splits) if item == 'val']
        test_ids = [idx for idx, item in enumerate(self.data.splits) if item == 'test']
        
        # Create the subsets
        self.train_set = Subset(self.data, train_ids)
        self.val_set = Subset(self.data, val_ids)
        self.test_set = Subset(self.data, test_ids)
        
    def _portion_split(self):
        
        # Get the cutoff positions
        cutoff_pos1 = int(len(self.data) * (1 - self.val_size - self.test_size))
        cutoff_pos2 = int(len(self.data) * (1 - self.test_size))
        
        # Create the subsets
        self.train_set = Subset(self.data, range(cutoff_pos1))
        self.val_set = Subset(self.data, range(cutoff_pos1, cutoff_pos2))
        self.test_set = Subset(self.data, range(cutoff_pos2, len(self.data)))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True, num_workers=self.num_workers, pin_memory = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size = self.batch_size, shuffle = False, num_workers=self.num_workers, pin_memory = True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.batch_size, shuffle = False, num_workers=self.num_workers, pin_memory = True)
    

class inputDataset(Dataset):
    def __init__(self, data:list[dict], device:torch.device = None):
        
        # Check if the device is provided
        device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract the texts
        self.texts = np.array([item['doc'] for item in data])
        
        # Extract inputs, and move them to the GPU
        self.inputs = [item['embeddings'] for item in data]
        
        # Extract the targets, and move them to the GPU
        self.targets = [item['vsa'].bfloat16() for item in data]
        
        #print('INPUTS:', np.unique([str(item.device) for item in self.inputs]))
        #print('TARGETS:', np.unique([str(item.device) for item in self.targets]))

        # Checks
        assert len(self.inputs) == len(self.targets), f"The number of inputs ({len(self.inputs)}) and targets ({len(self.targets)}) must be the same"
        
        # Save the parameters
        self.domains = [item['domain'] for item in data] if 'domain' in data[0] else None
        self.splits = [item['split'] for item in data] if 'split' in data[0] else None

        print(f'\n[INFO] Number of inputs: {len(self.inputs)}\n')
        
    def get_device(self):
        return self.inputs[0].device
        
    def get_input_dim(self):
        return self.inputs[0].shape[0]
    
    def get_target_dim(self):
        return self.targets[0].shape[0]

    def _standardize(self, tensor):
        return (tensor - tensor.mean()) / tensor.std()
    
    def _normalize(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def add_domains(inputs):
    
    # Load the all texts
    with open(path.join('data', 'features.json'), 'r') as file: 
        feature_domains = json.load(file)
        
    out = list()
    for item in inputs:
        
        # Split the document into items
        items = re.split(r"\s*[:=]\s*", item['doc'].strip())
        
        # Get the domains
        domains = [feature_domains[item] for item in items if item in feature_domains.keys()]

        # Check if there are domains
        if not domains:
            continue
        
        # Get the common domain
        common_domain = set(domains[0]).intersection(*domains[1:])
        common_domain = list(common_domain)[0] if len(common_domain) == 1 else domains[-1][0]
        
        # Add the domain
        item['domain'] = common_domain
        out.append(item)
    
    return out


def add_splits(item, splitted_docs):

    # Check if the document is in the splits
    if item['doc'].lower() in splitted_docs['train']:
        dataset = 'train'
    elif item['doc'].lower() in splitted_docs['val']:
        dataset = 'val'
    elif item['doc'].lower() in splitted_docs['test']:
        dataset = 'test'
    elif len(item['doc'].split()) == 1:
        dataset = 'train'
    else:
        return None
        #raise ValueError(f"Document not found in the splits: {item['doc']}")

    return dataset

def add_target_keys(item, pairs):
    
    # Split the document into items
    tokens = re.split(r"\s*[:=]\s*", item['doc'].strip())
    
    if len(tokens) == 1:
        return []
    
    # Get the pair
    target = tokens[-1]
    target_keys = pairs.get(target)
    
    # Check if the pair is found
    if target_keys:
        target_keys = [key for key in target_keys if key in tokens]
        if target_keys:
            target_keys.sort(key=lambda k: tokens.index(k))
            pair = [target, target_keys[-1]]
        else:
            pair = None
    else:
        raise ValueError(f"Pair not found for item: {target} ({item['doc']} --> {tokens})")

    return pair

def add_info_inputs(inputs, codebook):
    
    # Save the codebook items
    codebook_set = set(codebook.index)
    
    # Load the target pairs
    with open(path.join('data', 'pairs.json'), 'r') as f:
        pairs = json.load(f)
    
    # Load the splitted data
    with open(path.join('data', 'splitted_data.json'), 'r') as f:
        splitted_docs = json.load(f)
    splitted_docs = {setName: [doc.lower() for doc in docs] for setName, docs in splitted_docs.items()}
    
    # Load the VSA encodings
    vsa_encodings_path = path.join('outputs', 'hyperprobe', '_vsa.pickle')
    vsa_encodings = None
    if path.exists(vsa_encodings_path):
        with open(vsa_encodings_path, 'rb') as file:
            vsa_encodings = pickle.load(file)
            
    #print("vsa_encodings:", vsa_encodings)
    
    # "Bangkok : Thailand = Oslo : Norway"
    
    out = []
    for item in inputs: 
        
        # Attach the target keys
        item['concepts'] = add_target_keys(item, pairs)
        
        # Attach dataset splits
        item['split'] = add_splits(item, splitted_docs)
        
        # Process the embeddings
        if item['embeddings'].ndim > 1:
            item['embeddings'] = item['embeddings'].sum(dim=0)
        
        # Randomly permute the embeddings
        #item['embeddings'] = item['embeddings'][torch.randperm(item['embeddings'].shape[0])]
        
        # Attach the VSA encoding
        if vsa_encodings is not None:
            item['vsa'] = vsa_encodings.get(item.get('doc'))
        else:
            item['vsa'] = create_vsa_encodings(item, codebook, codebook_set, verbose = False)
        
        # Add the item to the output list     
        if item['concepts'] is not None and item['split'] is not None and item['vsa'] is not None:
            out.append(item)
        else:
            print('\n[WARNING] The item (item["doc"]) was not added to the output list.\n', 'CONCEPTS:', item.get('concepts'), 'SPLIT:', item.get('split'), 'VSA:', 'YES' if item.get('vsa') is not None else 'NO')
    
    # Save the VSA encodings
    if vsa_encodings is None:
        computed_vsa = {item['doc']: item['vsa'] for item in out}
        #with open(vsa_encodings_path, 'wb') as file:
        #    pickle.dump(computed_vsa, file, protocol=pickle.HIGHEST_PROTOCOL)

    return out

def add_info_QAinputs(inputs, codebook):
    
    # Save the codebook items
    codebook_set = set(codebook.index)
    
    # Load the VSA encodings
    vsa_encodings_path = path.join('outputs', 'hyperprobe', '_vsaQA.pickle')
    vsa_encodings = None
    if path.exists(vsa_encodings_path):
        with open(vsa_encodings_path, 'rb') as file:
            vsa_encodings = pickle.load(file)
    
    # Load the features        
    with open(path.join('data', 'squad', 'squad_training.json'), 'r') as file:
        qa_items = json.load(file)
    qa_items = {item['doc']: [f.lower() for f in item['features']] for item in qa_items}
    
    # Generate the random splits
    random_splits = np.random.default_rng(seed = 101).choice(['train','val','test'], size=len(inputs), p=[.70, .15, .15])

    out = []
    for idk, item in enumerate(inputs): 
        
        # Attach dataset splits
        item['split'] = random_splits[idk]
        
        # Attach the features
        item['concepts'] = qa_items.get(item['doc'].strip())
            
        # Process the embeddings
        if item['embeddings'].ndim > 1:
            item['embeddings'] = item['embeddings'].sum(dim=0)
        
        # Attach the VSA encoding
        if vsa_encodings is not None:
            item['vsa'] = vsa_encodings.get(item['doc'])
        else:
            item['vsa'] = create_vsa_encodings_QA(item, codebook, codebook_set)
            
        # Add the item to the output list     
        if item.get('concepts') is not None and item.get('split') is not None and item.get('vsa') is not None:
            out.append(item)
        else:
            print(f"\n[WARNING] The item ({item['doc']}) was not added to the output list:", 'CONCEPTS:', item.get('concepts'), 'SPLIT:', item.get('split'), 'VSA:', item.get('vsa'))
            
    # Save the VSA encodings
    #if vsa_encodings is None:
        #computed_vsa = {item['doc']: item['vsa'] for item in out}
        #with open(vsa_encodings_path, 'wb') as file:
            #pickle.dump(computed_vsa, file, protocol=pickle.HIGHEST_PROTOCOL)

    return out

def create_vsa_encodings_QA(item, codebook, codebook_set):
    if not item['concepts']:
        return None
    
    features = codebook.loc[list(item['concepts'])].values
    encoding = torchhd.multiset(torchhd.MAPTensor(features)).normalize()

    return encoding.as_subclass(torch.Tensor).to(torch.int8)