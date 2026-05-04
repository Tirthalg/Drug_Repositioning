"""
Drug Repositioning Pipeline with GNN - IMPROVED VERSION
========================================================
All improvements applied:
  - P0: Degree feature bug fix, prediction decoder fix
  - P1: Interaction features, Focal Loss alpha fix
  - P2: LayerNorm, residual connections, LeakyReLU, larger dims, gradient clipping
  - P3: Removed duplicate cells, removed AddSelfLoops, fixed comments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, balanced_accuracy_score
)
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("=" * 80)
print("STEP 1: Loading Data")
print("=" * 80)

with open('bio_graph_raw.pkl', 'rb') as f:
    G = pickle.load(f)

# Separate node types
chemicals = sorted([n for n, d in G.nodes(data=True) if d['node_type'] == 'chemical'])
diseases  = sorted([n for n, d in G.nodes(data=True) if d['node_type'] == 'disease'])
genes     = sorted([n for n, d in G.nodes(data=True) if d['node_type'] == 'gene'])

chem_idx = {c: i for i, c in enumerate(chemicals)}
disease_idx = {d: i for i, d in enumerate(diseases)}
gene_idx = {g: i for i, g in enumerate(genes)}

print(f"Number of chemicals: {len(chemicals)}")
print(f"Number of diseases: {len(diseases)}")
print(f"Number of genes: {len(genes)}")

# Extract features
chem_features = torch.tensor(
    np.array([G.nodes[c]['x'] for c in chemicals], dtype=np.float32)
)
disease_features = torch.tensor(
    np.array([G.nodes[d]['x'] for d in diseases], dtype=np.float32)
)
gene_features = torch.tensor(
    np.array([G.nodes[g]['x'] for g in genes], dtype=np.float32)
)

print(f"Chemical feature dim: {chem_features.shape[1]}")
print(f"Disease feature dim: {disease_features.shape[1]}")
print(f"Gene feature dim: {gene_features.shape[1]}")


# ============================================================================
# 2. EDGE EXTRACTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Extracting Edges")
print("=" * 80)

chem_gene_edges = []
chem_disease_edges = []
gene_disease_edges = []

for u, v, data in G.edges(data=True):
    if data['edge_type'] == 'chem_gene':
        chem_gene_edges.append([chem_idx[u], gene_idx[v]])
    elif data['edge_type'] == 'chem_disease':
        chem_disease_edges.append([chem_idx[u], disease_idx[v]])
    elif data['edge_type'] == 'gene_disease':
        gene_disease_edges.append([gene_idx[u], disease_idx[v]])

chem_gene_edges = torch.tensor(chem_gene_edges).t().contiguous()
chem_disease_edges = torch.tensor(chem_disease_edges).t().contiguous()
gene_disease_edges = torch.tensor(gene_disease_edges).t().contiguous()

print(f"Chemical-Gene edges: {chem_gene_edges.shape[1]}")
print(f"Chemical-Disease edges: {chem_disease_edges.shape[1]}")
print(f"Gene-Disease edges: {gene_disease_edges.shape[1]}")

# --------------------------------------------------------------------------
# [FIX P0] Compute node degrees FIRST, then append log-degree features
# --------------------------------------------------------------------------
print("\nComputing node degrees for hard negative sampling...")
chem_degrees = torch.zeros(len(chemicals), dtype=torch.long)
disease_degrees = torch.zeros(len(diseases), dtype=torch.long)
gene_degrees = torch.zeros(len(genes), dtype=torch.long)  # [FIX] Added gene degrees

# Compute chem-disease degrees
for i in range(chem_disease_edges.shape[1]):
    chem_degrees[chem_disease_edges[0, i]] += 1
    disease_degrees[chem_disease_edges[1, i]] += 1

# Compute gene degrees from chem-gene and gene-disease edges
for i in range(chem_gene_edges.shape[1]):
    gene_degrees[chem_gene_edges[1, i]] += 1
for i in range(gene_disease_edges.shape[1]):
    gene_degrees[gene_disease_edges[0, i]] += 1

# [FIX P0] NOW compute log-degree features (after degrees are computed)
chem_log_deg = torch.log1p(chem_degrees.float()).unsqueeze(1)
disease_log_deg = torch.log1p(disease_degrees.float()).unsqueeze(1)
gene_log_deg = torch.log1p(gene_degrees.float()).unsqueeze(1)  # [FIX] Gene degree features

chem_features = torch.cat([chem_features, chem_log_deg], dim=1)
disease_features = torch.cat([disease_features, disease_log_deg], dim=1)
gene_features = torch.cat([gene_features, gene_log_deg], dim=1)  # [FIX] Consistent

print(f"Chemical degree stats - Min: {chem_degrees.min()}, Max: {chem_degrees.max()}, Mean: {chem_degrees.float().mean():.2f}, Median: {chem_degrees.float().median():.2f}")
print(f"Disease degree stats - Min: {disease_degrees.min()}, Max: {disease_degrees.max()}, Mean: {disease_degrees.float().mean():.2f}, Median: {disease_degrees.float().median():.2f}")
print(f"Gene degree stats - Min: {gene_degrees.min()}, Max: {gene_degrees.max()}, Mean: {gene_degrees.float().mean():.2f}, Median: {gene_degrees.float().median():.2f}")
print(f"Updated feature dims - Chemical: {chem_features.shape[1]}, Disease: {disease_features.shape[1]}, Gene: {gene_features.shape[1]}")


# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Splitting Data (Train/Val/Test)")
print("=" * 80)

num_edges = chem_disease_edges.shape[1]
perm = torch.randperm(num_edges)

# 70% train, 15% val, 15% test
train_size = int(0.7 * num_edges)
val_size = int(0.15 * num_edges)

train_idx = perm[:train_size]
val_idx = perm[train_size:train_size + val_size]
test_idx = perm[train_size + val_size:]

train_edges = chem_disease_edges[:, train_idx]
val_edges = chem_disease_edges[:, val_idx]
test_edges = chem_disease_edges[:, test_idx]

print(f"Train edges: {train_edges.shape[1]}")
print(f"Val edges: {val_edges.shape[1]}")
print(f"Test edges: {test_edges.shape[1]}")


# ============================================================================
# 4. NEGATIVE SAMPLING (Degree-Preserving Hard Negatives)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Generating Negative Samples")
print("=" * 80)

def create_degree_bins(degrees, bin_edges):
    """
    Assign nodes to degree bins based on their degrees.
    Returns a dictionary: bin_id -> list of node indices
    """
    bins = {i: [] for i in range(len(bin_edges) - 1)}

    for node_id, deg in enumerate(degrees.tolist()):
        for bin_id in range(len(bin_edges) - 1):
            if bin_edges[bin_id] <= deg < bin_edges[bin_id + 1]:
                bins[bin_id].append(node_id)
                break
        else:  # Handle nodes with degree >= max bin edge
            bins[len(bin_edges) - 2].append(node_id)

    return bins


def degree_preserving_negative_sampling(edge_index, chem_degrees, disease_degrees,
                                       num_neg_samples, bin_edges_chem, bin_edges_disease):
    """
    Generate hard negative samples that match the degree distribution of positive edges.
    """
    pos_edges_set = set(map(tuple, edge_index.t().tolist()))

    chem_bins = create_degree_bins(chem_degrees, bin_edges_chem)
    disease_bins = create_degree_bins(disease_degrees, bin_edges_disease)

    pos_chem_degrees = chem_degrees[edge_index[0]].numpy()
    pos_disease_degrees = disease_degrees[edge_index[1]].numpy()

    neg_edges = []
    neg_chem_degrees = []
    neg_disease_degrees = []

    attempts = 0
    max_attempts = num_neg_samples * 20
    num_pos_edges = edge_index.shape[1]

    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        pos_idx = random.randint(0, num_pos_edges - 1)
        target_chem_deg = chem_degrees[edge_index[0, pos_idx]].item()
        target_disease_deg = disease_degrees[edge_index[1, pos_idx]].item()

        chem_bin_id = None
        for bin_id in range(len(bin_edges_chem) - 1):
            if bin_edges_chem[bin_id] <= target_chem_deg < bin_edges_chem[bin_id + 1]:
                chem_bin_id = bin_id
                break
        if chem_bin_id is None:
            chem_bin_id = len(bin_edges_chem) - 2

        disease_bin_id = None
        for bin_id in range(len(bin_edges_disease) - 1):
            if bin_edges_disease[bin_id] <= target_disease_deg < bin_edges_disease[bin_id + 1]:
                disease_bin_id = bin_id
                break
        if disease_bin_id is None:
            disease_bin_id = len(bin_edges_disease) - 2

        chem_candidates = chem_bins[chem_bin_id].copy()
        disease_candidates = disease_bins[disease_bin_id].copy()

        if not chem_candidates:
            for offset in [1, -1, 2, -2]:
                neighbor_bin = chem_bin_id + offset
                if 0 <= neighbor_bin < len(bin_edges_chem) - 1 and chem_bins[neighbor_bin]:
                    chem_candidates = chem_bins[neighbor_bin].copy()
                    break

        if not disease_candidates:
            for offset in [1, -1, 2, -2]:
                neighbor_bin = disease_bin_id + offset
                if 0 <= neighbor_bin < len(bin_edges_disease) - 1 and disease_bins[neighbor_bin]:
                    disease_candidates = disease_bins[neighbor_bin].copy()
                    break

        if chem_candidates and disease_candidates:
            src = random.choice(chem_candidates)
            dst = random.choice(disease_candidates)

            if (src, dst) not in pos_edges_set:
                neg_edges.append([src, dst])
                neg_chem_degrees.append(chem_degrees[src].item())
                neg_disease_degrees.append(disease_degrees[dst].item())

        attempts += 1

    if len(neg_edges) < num_neg_samples:
        print(f"  Warning: Could only generate {len(neg_edges)} negative samples (requested {num_neg_samples})")

    print(f"  Positive edges - Chem degree mean: {pos_chem_degrees.mean():.2f}, Disease degree mean: {pos_disease_degrees.mean():.2f}")
    if neg_chem_degrees:
        print(f"  Negative edges - Chem degree mean: {np.mean(neg_chem_degrees):.2f}, Disease degree mean: {np.mean(neg_disease_degrees):.2f}")

    return torch.tensor(neg_edges).t().contiguous() if neg_edges else torch.empty((2, 0), dtype=torch.long)


# Define degree bins
bin_edges_chem = [0, 10, 50, 100, 500, 2000, float('inf')]
bin_edges_disease = [0, 10, 50, 100, 500, 2000, float('inf')]

print(f"Using degree bins for chemicals: {bin_edges_chem[:-1]}")
print(f"Using degree bins for diseases: {bin_edges_disease[:-1]}")

print("\nGenerating HARD negative samples for training set...")
print("  (excluding only train edges, matching degree distribution)")
train_neg_edges = degree_preserving_negative_sampling(
    train_edges, chem_degrees, disease_degrees,
    train_edges.shape[1], bin_edges_chem, bin_edges_disease
)

print("\nGenerating HARD negative samples for validation set...")
print("  (excluding train + val edges, matching degree distribution)")
train_val_edges = torch.cat([train_edges, val_edges], dim=1)
val_neg_edges = degree_preserving_negative_sampling(
    train_val_edges, chem_degrees, disease_degrees,
    val_edges.shape[1], bin_edges_chem, bin_edges_disease
)

print("\nGenerating HARD negative samples for test set...")
print("  (excluding train + val + test edges, matching degree distribution)")
all_edges = torch.cat([train_edges, val_edges, test_edges], dim=1)
test_neg_edges = degree_preserving_negative_sampling(
    all_edges, chem_degrees, disease_degrees,
    test_edges.shape[1], bin_edges_chem, bin_edges_disease
)

print(f"Train negative edges: {train_neg_edges.shape[1]}")
print(f"Val negative edges: {val_neg_edges.shape[1]}")
print(f"Test negative edges: {test_neg_edges.shape[1]}")


# ============================================================================
# 5. BUILD HETEROGENEOUS GRAPH
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Building Heterogeneous Graph")
print("=" * 80)

data = HeteroData()

data['chemical'].x = chem_features
data['disease'].x = disease_features
data['gene'].x = gene_features

data['chemical', 'chem_gene', 'gene'].edge_index = chem_gene_edges
data['gene', 'gene_disease', 'disease'].edge_index = gene_disease_edges

# Use only TRAINING chem-disease edges for message passing
data['chemical', 'chem_disease', 'disease'].edge_index = train_edges

# Store edge labels for link prediction
data['chemical', 'chem_disease', 'disease'].edge_label_index = torch.cat([train_edges, train_neg_edges], dim=1)
data['chemical', 'chem_disease', 'disease'].edge_label = torch.cat([
    torch.ones(train_edges.shape[1]),
    torch.zeros(train_neg_edges.shape[1])
])

# [FIX P3] Make undirected but do NOT add self-loops (can hurt heterogeneous GNN)
data = ToUndirected()(data)

print(data)


# ============================================================================
# 6. MINI-BATCH DATA LOADERS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Creating Mini-Batch Data Loaders")
print("=" * 80)

# [FIX] Increased batch size from 512 -> 1024 for better gradient stability
BATCH_SIZE = 1024
# Neighbor sampling: 25 in 1st hop, 10 in 2nd hop, 5 in 3rd hop
NUM_NEIGHBORS = [25, 10, 5]

train_loader = LinkNeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=(('chemical', 'chem_disease', 'disease'),
                      torch.cat([train_edges, train_neg_edges], dim=1)),
    edge_label=torch.cat([
        torch.ones(train_edges.shape[1]),
        torch.zeros(train_neg_edges.shape[1])
    ]),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# Validation loader
val_edge_label_index = torch.cat([val_edges, val_neg_edges], dim=1)
val_edge_label = torch.cat([
    torch.ones(val_edges.shape[1]),
    torch.zeros(val_neg_edges.shape[1])
])

val_loader = LinkNeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=(('chemical', 'chem_disease', 'disease'), val_edge_label_index),
    edge_label=val_edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Test loader
test_edge_label_index = torch.cat([test_edges, test_neg_edges], dim=1)
test_edge_label = torch.cat([
    torch.ones(test_edges.shape[1]),
    torch.zeros(test_neg_edges.shape[1])
])

test_loader = LinkNeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=(('chemical', 'chem_disease', 'disease'), test_edge_label_index),
    edge_label=test_edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# ============================================================================
# 7. IMPROVED MODEL DEFINITION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Defining IMPROVED Heterogeneous GNN Model")
print("=" * 80)


class HeteroGNN(nn.Module):
    """
    Improved Heterogeneous Graph Neural Network
    Changes from original:
      - Larger hidden/output dims (256/128)
      - LayerNorm after each conv layer
      - Residual connections between layers
      - LeakyReLU instead of ReLU
      - MLP decoder with interaction features [src, dst, src*dst]
    """

    def __init__(self, hidden_channels=256, out_channels=128, num_layers=3, dropout=0.2):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # GNN conv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ('chemical', 'chem_gene', 'gene'): SAGEConv((-1, -1), hidden_channels),
                ('gene', 'rev_chem_gene', 'chemical'): SAGEConv((-1, -1), hidden_channels),
                ('chemical', 'chem_disease', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'rev_chem_disease', 'chemical'): SAGEConv((-1, -1), hidden_channels),
                ('gene', 'gene_disease', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'rev_gene_disease', 'gene'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean'))  # Try 'sum' as an alternative if mean underperforms

        # [FIX P2] LayerNorm per node type per layer
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.ModuleDict({
                'chemical': nn.LayerNorm(hidden_channels),
                'disease': nn.LayerNorm(hidden_channels),
                'gene': nn.LayerNorm(hidden_channels),
            }))

        # [FIX P2] Projection layers for residual connections (first layer input -> hidden)
        self.input_projs = nn.ModuleDict({
            'chemical': Linear(-1, hidden_channels),
            'disease': Linear(-1, hidden_channels),
            'gene': Linear(-1, hidden_channels),
        })

        # Final projection layers
        self.lin_chemical = Linear(hidden_channels, out_channels)
        self.lin_disease = Linear(hidden_channels, out_channels)

        # [FIX P1] MLP Decoder with interaction features: [src, dst, src*dst]
        # Input is 3 * out_channels instead of 2 * out_channels
        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 3, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        # Project inputs to hidden_channels for residual connections
        h_dict = {k: self.input_projs[k](x) for k, x in x_dict.items()}

        for i, conv in enumerate(self.convs):
            # Store for residual
            prev_dict = h_dict

            # Message passing
            h_dict = conv(h_dict, edge_index_dict)

            # [FIX P2] LayerNorm + LeakyReLU + Dropout + Residual
            new_h_dict = {}
            for node_type in h_dict:
                h = self.norms[i][node_type](h_dict[node_type])
                h = F.leaky_relu(h, negative_slope=0.2)
                h = F.dropout(h, p=self.dropout, training=self.training)

                # Residual connection (add previous layer's output)
                if node_type in prev_dict and prev_dict[node_type].shape == h.shape:
                    h = h + prev_dict[node_type]

                new_h_dict[node_type] = h

            h_dict = new_h_dict

        z_chemical = self.lin_chemical(h_dict['chemical'])
        z_disease = self.lin_disease(h_dict['disease'])

        return z_chemical, z_disease

    def decode(self, z_chemical, z_disease, edge_label_index):
        src = z_chemical[edge_label_index[0]]
        dst = z_disease[edge_label_index[1]]

        # [FIX P1] Use interaction features: [src, dst, src*dst]
        interaction = src * dst
        edge_feat = torch.cat([src, dst, interaction], dim=-1)
        return self.decoder(edge_feat).squeeze()


# Initialize improved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGNN().to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")


# ============================================================================
# 8. TRAINING SETUP
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Setting up Training (Improved)")
print("=" * 80)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
)

# [FIX P1] Focal Loss with alpha=0.5 for balanced dataset
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

criterion = FocalLoss()


# ============================================================================
# 9. TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one epoch using mini-batches"""
    model.train()
    total_loss = 0
    total_examples = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        z_chemical, z_disease = model(batch.x_dict, batch.edge_index_dict)

        # Get edge predictions for this batch
        edge_label_index = batch['chemical', 'chem_disease', 'disease'].edge_label_index
        edge_label = batch['chemical', 'chem_disease', 'disease'].edge_label

        # Compute scores
        pred = model.decode(z_chemical, z_disease, edge_label_index)

        # Compute loss
        loss = criterion(pred, edge_label)

        # Backward pass
        loss.backward()

        # [FIX P2] Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * edge_label.size(0)
        total_examples += edge_label.size(0)

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model using mini-batches"""
    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)

        # Forward pass
        z_chemical, z_disease = model(batch.x_dict, batch.edge_index_dict)

        # Get edge predictions for this batch
        edge_label_index = batch['chemical', 'chem_disease', 'disease'].edge_label_index
        edge_label = batch['chemical', 'chem_disease', 'disease'].edge_label

        # Compute scores
        pred = model.decode(z_chemical, z_disease, edge_label_index)
        pred = torch.sigmoid(pred)

        all_preds.append(pred.cpu())
        all_labels.append(edge_label.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    return auc, ap


# ============================================================================
# 10. TRAINING LOOP (with Early Stopping)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Training the IMPROVED Model")
print("=" * 80)

num_epochs = 50
best_val_auc = 0
patience = 15
patience_counter = 0
losses = []
rocs = []

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_auc, val_ap = evaluate(model, val_loader, device)

    scheduler.step(val_auc)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:03d} | '
          f'Loss: {train_loss:.4f} | '
          f'Val AUC: {val_auc:.4f} | '
          f'Val AP: {val_ap:.4f} | '
          f'LR: {current_lr:.6f}')

    losses.append(train_loss)
    rocs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_minibatch.pt')
        print("  → New best model saved!")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'\nEarly stopping at epoch {epoch}')
        break

print(f'\nBest validation AUC: {best_val_auc:.4f}')


# ============================================================================
# 11. FINAL EVALUATION (Full Metrics)
# ============================================================================

@torch.no_grad()
def evaluate_full(model, loader, device, threshold=0.5, top_k=100):
    """
    Full evaluation with multiple metrics
    """
    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(loader):
        batch = batch.to(device)

        z_chemical, z_disease = model(batch.x_dict, batch.edge_index_dict)

        edge_label_index = batch['chemical', 'chem_disease', 'disease'].edge_label_index
        edge_label = batch['chemical', 'chem_disease', 'disease'].edge_label

        pred = model.decode(z_chemical, z_disease, edge_label_index)
        pred = torch.sigmoid(pred)

        all_preds.append(pred.cpu())
        all_labels.append(edge_label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Threshold-based metrics
    binary_preds = (all_preds >= threshold).astype(int)

    acc = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    bal_acc = balanced_accuracy_score(all_labels, binary_preds)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    specificity = tn / (tn + fp)

    # Ranking metrics
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    # Precision@K / Recall@K
    sorted_indices = np.argsort(-all_preds)
    top_k_indices = sorted_indices[:top_k]
    top_k_labels = all_labels[top_k_indices]
    precision_at_k = np.sum(top_k_labels) / top_k
    recall_at_k = np.sum(top_k_labels) / np.sum(all_labels)

    # Curves
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)

    results = {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "PR-AUC": ap,
        f"Precision@{top_k}": precision_at_k,
        f"Recall@{top_k}": recall_at_k,
        "Confusion Matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }

    return results, fpr, tpr, precision_curve, recall_curve


print("\n" + "=" * 80)
print("FINAL TEST EVALUATION (Full Metrics)")
print("=" * 80)

model.load_state_dict(torch.load('best_model_minibatch.pt'))

results, fpr, tpr, pr_curve, rc_curve = evaluate_full(
    model,
    test_loader,
    device,
    threshold=0.5,
    top_k=200
)

for k, v in results.items():
    if k != "Confusion Matrix":
        print(f"{k}: {v:.4f}")
    else:
        print("\nConfusion Matrix:")
        for key, val in v.items():
            print(f"  {key}: {val}")


# ============================================================================
# 12. PREDICTION FUNCTION (Memory-Efficient, uses MLP decoder)
# ============================================================================

@torch.no_grad()
def predict_top_k_batch(model, data, chemicals_list, diseases_list, k=10, batch_size=100, device='cpu'):
    """
    Memory-efficient prediction using batching.
    [FIX P0] Uses the trained MLP decoder instead of raw dot product.
    """
    model.eval()

    full_data = data.to(device)
    z_chemical, z_disease = model(full_data.x_dict, full_data.edge_index_dict)

    num_diseases = z_disease.shape[0]
    disease_idx_to_name = {v: k for k, v in diseases_list.items()}

    predictions = {}
    chem_items = list(chemicals_list.items())

    for i in tqdm(range(0, len(chem_items), batch_size), desc="Predicting"):
        batch_chems = chem_items[i:i+batch_size]

        for chem_name, chem_id in batch_chems:
            # [FIX P0] Use MLP decoder for scoring (consistent with training)
            edge_label_index = torch.stack([
                torch.full((num_diseases,), chem_id, dtype=torch.long, device=device),
                torch.arange(num_diseases, device=device)
            ])
            scores = model.decode(z_chemical, z_disease, edge_label_index)
            scores = torch.sigmoid(scores)

            # Get top-k diseases
            top_k_scores, top_k_indices = torch.topk(scores, k)

            top_diseases = [(disease_idx_to_name[idx.item()], score.item())
                           for idx, score in zip(top_k_indices, top_k_scores)]

            predictions[chem_name] = top_diseases

    return predictions


# Example predictions
print("\n" + "=" * 80)
print("STEP 11: Generating Predictions (Example)")
print("=" * 80)

first_5_chems = {k: v for k, v in list(chem_idx.items())[:5]}
predictions = predict_top_k_batch(model, data, first_5_chems, disease_idx, k=10, device=device)

print("\nTop-10 Drug-Disease Predictions for First 5 Chemicals:")
print("=" * 80)
for chem, disease_scores in predictions.items():
    print(f"\n{chem}:")
    for i, (disease, score) in enumerate(disease_scores, 1):
        print(f"  {i}. {disease}: {score:.4f}")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print("\nImprovements applied:")
print("- [P0] Fixed degree features (computed AFTER degrees, not before)")
print("- [P0] Fixed prediction to use MLP decoder (matches training)")
print("- [P1] Added interaction features [src, dst, src*dst] in decoder")
print("- [P1] Focal Loss alpha=0.5 for balanced dataset")
print("- [P2] Added LayerNorm after each GNN layer")
print("- [P2] Added residual connections between layers")
print("- [P2] LeakyReLU instead of ReLU")
print("- [P2] Larger model: hidden=256, output=128")
print("- [P2] Added gene degree features for consistency")
print("- [P2] Gradient clipping (max_norm=1.0)")
print("- [P2] Increased batch size to 1024")
print("- [P3] Removed AddSelfLoops (can hurt heterogeneous GNNs)")
print("- [P3] Removed duplicate cells")
print("- [P3] Fixed comments to match actual config")
