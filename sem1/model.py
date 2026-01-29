import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader  # Changed to NeighborLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from scipy.special import expit  # Stable sigmoid
import numpy as np
import os
import pickle
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import AddSelfLoops, ToUndirected
with open('grph.pickle', 'rb') as f:
    G = pickle.load(f)

chemicals = sorted([n for n, d in G.nodes(data=True) if d['node_type'] == 'chemical'])
diseases = sorted([n for n, d in G.nodes(data=True) if d['node_type'] == 'disease'])
genes = sorted([n for n, d in G.nodes(data=True) if d['node_type'] == 'gene'])

chem_idx = {c: i for i, c in enumerate(chemicals)}
disease_idx = {d: i for i, d in enumerate(diseases)}
gene_idx = {g: i for i, g in enumerate(genes)}


# Extract node embeddings
chem_features = np.array([G.nodes[c]['embedding'] for c in chemicals], dtype=np.float32)
disease_features = np.array([G.nodes[d]['embedding'] for d in diseases], dtype=np.float32)
gene_features = np.array([G.nodes[g]['embedding'] for g in genes], dtype=np.float32)

# Extract edges
chem_gene_edges = []
chem_disease_edges = []
gene_disease_edges = []
for u, v, data in G.edges(data=True):
    edge_type = data['edge_type']
    if edge_type == 'chem_gene':
        chem_gene_edges.append([chem_idx[u], gene_idx[v]])
    elif edge_type == 'chem_disease':
        chem_disease_edges.append([chem_idx[u], disease_idx[v]])
    elif edge_type == 'gene_disease':
        gene_disease_edges.append([gene_idx[u], disease_idx[v]])


# Convert to tensors
chem_gene_edges = torch.tensor(chem_gene_edges, dtype=torch.long).t()
chem_disease_edges = torch.tensor(chem_disease_edges, dtype=torch.long).t()
gene_disease_edges = torch.tensor(gene_disease_edges, dtype=torch.long).t()


max_dim = max(chem_features.shape[1], disease_features.shape[1], gene_features.shape[1])
chem_features = np.pad(chem_features, ((0, 0), (0, max_dim - chem_features.shape[1])), mode='constant')
disease_features = np.pad(disease_features, ((0, 0), (0, max_dim - disease_features.shape[1])), mode='constant')
gene_features = np.pad(gene_features, ((0, 0), (0, max_dim - gene_features.shape[1])), mode='constant')


data = HeteroData()
data['chemical'].x = torch.tensor(chem_features, dtype=torch.float)
data['disease'].x = torch.tensor(disease_features, dtype=torch.float)
data['gene'].x = torch.tensor(gene_features, dtype=torch.float)
data['chemical', 'chem_gene', 'gene'].edge_index = chem_gene_edges
data['chemical', 'chem_disease', 'disease'].edge_index = chem_disease_edges
data['gene', 'gene_disease', 'disease'].edge_index = gene_disease_edges
print(data)

data = ToUndirected()(data)
data = AddSelfLoops()(data)


if isinstance(data, HeteroData):
    for edge_type in data.edge_types:
        if 'edge_index' in data[edge_type]:
            data[edge_type].edge_index = data[edge_type].edge_index.contiguous()

# If 'data' is a homogeneous Data object (for completeness, though your code suggests hetero)
else:
    data.edge_index = data.edge_index.contiguous()


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        # Define HeteroConv for each edge type
        self.conv1 = HeteroConv({
            ('chemical', 'chem_gene', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('chemical', 'chem_disease', 'disease'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'gene_disease', 'disease'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'rev_chem_gene', 'chemical'): SAGEConv((-1, -1), hidden_channels),
            ('disease', 'rev_chem_disease', 'chemical'): SAGEConv((-1, -1), hidden_channels),
            ('disease', 'rev_gene_disease', 'gene'): SAGEConv((-1, -1), hidden_channels)
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('chemical', 'chem_gene', 'gene'): SAGEConv(hidden_channels, hidden_channels),
            ('chemical', 'chem_disease', 'disease'): SAGEConv(hidden_channels, hidden_channels),
            ('gene', 'gene_disease', 'disease'): SAGEConv(hidden_channels, hidden_channels),
            ('gene', 'rev_chem_gene', 'chemical'): SAGEConv(hidden_channels, hidden_channels),
            ('disease', 'rev_chem_disease', 'chemical'): SAGEConv(hidden_channels, hidden_channels),
            ('disease', 'rev_gene_disease', 'gene'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

def predict_edges(drug_emb, disease_emb, edge_index):
    row, col = edge_index
    return (drug_emb[row] * disease_emb[col]).sum(dim=-1)
def load_model(model, optimizer, checkpoint_path, device):
    """Load model and optimizer state from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}, loss: {loss:.4f}")
    return model, optimizer

def evaluate(model, data, device, batch_size=128):
    """Evaluate the model on chemical-disease edge prediction using batched processing."""
    model.eval()
    all_scores = []
    all_labels = []
    
    # Use LinkNeighborLoader for batched evaluation (edge-centric)
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[100,50],
        batch_size=batch_size,
        edge_label_index=(('chemical', 'chem_disease', 'disease'), data['chemical', 'chem_disease', 'disease'].edge_index),
        shuffle=False
    )
    
    with torch.no_grad():
        i=100
        for batch in tqdm(loader):
            if i==0:
                break
            i-=1    
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            drug_emb = out['chemical']
            disease_emb = out['disease']
            
            pos_edge_index = batch['chemical', 'chem_disease', 'disease'].edge_index
            if pos_edge_index.size(1) == 0:
                continue
            pos_scores = predict_edges(drug_emb, disease_emb, pos_edge_index)
            pos_labels = torch.ones(pos_edge_index.size(1), device=device)
            
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(batch['chemical'].num_nodes, batch['disease'].num_nodes),
                num_neg_samples=pos_edge_index.size(1)
            )
            neg_scores = predict_edges(drug_emb, disease_emb, neg_edge_index)
            neg_labels = torch.zeros(neg_edge_index.size(1), device=device)
            
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    print(all_scores)
    print(all_labels)   
    # Stable sigmoid for probabilities
    probs = expit(all_scores)  # Replaces manual sigmoid to avoid overflow
    predictions = (probs > 0.5).astype(int)
    
    # Compute metrics, including balanced accuracy
    metrics = {
        'roc_auc': roc_auc_score(all_labels, probs),
        'precision': precision_score(all_labels, predictions, zero_division=0),
        'recall': recall_score(all_labels, predictions, zero_division=0),
        'f1': f1_score(all_labels, predictions, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(all_labels, predictions)
    }
    return metrics

def predict_top_drugs(model, data, disease_id, chem_idx, disease_idx, top_k=5, device='cpu', batch_size=128):
    """Predict top-k drugs for a given disease ID using batched processing."""
    model.eval()
    if disease_id not in disease_idx:
        raise ValueError(f"Disease ID {disease_id} not found in disease_idx.")
    
    disease_global_idx = disease_idx[disease_id]
    all_scores = []
    all_chem_indices = []
    
    # Use NeighborLoader for node-centric batching on chemicals
    loader = NeighborLoader(
        data,
        num_neighbors=[100,50],
        batch_size=batch_size,
        input_nodes=('chemical', torch.arange(data['chemical'].num_nodes)),  # All chemicals
        shuffle=False
    )
    
    with torch.no_grad():
        i = 1000
        for batch in tqdm(loader):
            
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            drug_emb = out['chemical']
            
            # Check if disease is in the batch subgraph (via node_id)
            if 'disease' in batch and disease_global_idx in batch['disease'].node_id:
                # Map global disease idx to batch-local idx
                local_disease_idx = (batch['disease'].node_id == disease_global_idx).nonzero(as_tuple=True)[0]
                if local_disease_idx.numel() > 0:
                    disease_emb_single = out['disease'][local_disease_idx].unsqueeze(0)
                    scores = (drug_emb * disease_emb_single).sum(dim=1).cpu().numpy()
                    batch_chem_globals = batch['chemical'].node_id.cpu().numpy()
                    all_scores.extend(scores)
                    all_chem_indices.extend(batch_chem_globals)
            
            # If disease not in batch, skip (or handle rarely with full forward if needed)
    
    if len(all_scores) == 0:
        raise ValueError(f"Disease {disease_id} not sampled in any batch. Increase num_neighbors or use full graph on CPU.")
    
    # Get top-k
    top_k_indices = np.argsort(all_scores)[::-1][:top_k]
    top_k_scores = [all_scores[i] for i in top_k_indices]
    top_k_chem_globals = [all_chem_indices[i] for i in top_k_indices]
    
    idx_to_chem = {i: c for c, i in chem_idx.items()}
    top_k_drugs = [(idx_to_chem[idx], score) for idx, score in zip(top_k_chem_globals, top_k_scores)]
    
    return top_k_drugs
# Example usage (with CPU fallback)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGNN(hidden_channels=64, metadata=data.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

checkpoint_path = 'checkpoints/model_epoch_002.pt'
model, optimizer = load_model(model, optimizer, checkpoint_path, device)

metrics = evaluate(model, data, device, batch_size=128)
print("Evaluation Metrics:")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")