"""
Complete Drug Repositioning using Heterogeneous GNN
Link Prediction on Chemical-Disease Interactions
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, Linear
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


class HeteroDrugGNN(torch.nn.Module):
    """
    Heterogeneous GNN for drug repositioning
    Uses message passing across all edge types and node types
    """
    def __init__(self, hidden_channels, out_channels, num_layers, metadata):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum'))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels, add_self_loops=False)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)
        
        # Last layer
        self.convs.append(HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum'))
        
    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:  # Apply activation to all but last layer
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict


class LinkPredictor(torch.nn.Module):
    """
    Simple MLP-based link predictor for chemical-disease edges
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2):
        super().__init__()
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        
    def forward(self, x_chemical, x_disease, edge_label_index):
        # Get source and target node embeddings
        src = x_chemical[edge_label_index[0]]
        dst = x_disease[edge_label_index[1]]
        
        # Combine embeddings (element-wise product + concatenation)
        x = src * dst  # Hadamard product
        
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i != len(self.lins) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        return x.squeeze(-1)


def split_data(data, edge_type=('chemical', 'chem_disease', 'disease'), 
               val_ratio=0.1, test_ratio=0.1):
    """
    Split edges into train/val/test sets
    """
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=1.0,  # 1:1 positive to negative ratio
        edge_types=[edge_type],
        rev_edge_types=[('disease', 'rev_chem_disease', 'chemical')],
        is_undirected=False,
    )
    
    train_data, val_data, test_data = transform(data)
    
    return train_data, val_data, test_data


def train_epoch(model, predictor, data, optimizer, device, edge_type):
    """
    Train for one epoch
    """
    model.train()
    predictor.train()
    optimizer.zero_grad()
    
    # Get node embeddings
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    # Get edge labels and indices for the target edge type
    edge_label_index = data[edge_type].edge_label_index
    edge_label = data[edge_type].edge_label
    
    # Predict
    pred = predictor(
        x_dict['chemical'],
        x_dict['disease'],
        edge_label_index
    )
    
    # Compute loss
    loss = F.binary_cross_entropy_with_logits(pred, edge_label)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, predictor, data, device, edge_type):
    """
    Evaluate the model
    """
    model.eval()
    predictor.eval()
    
    # Get node embeddings
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    # Get edge labels and indices
    edge_label_index = data[edge_type].edge_label_index
    edge_label = data[edge_type].edge_label
    
    # Predict
    pred = predictor(
        x_dict['chemical'],
        x_dict['disease'],
        edge_label_index
    )
    
    pred = torch.sigmoid(pred).cpu().numpy()
    edge_label = edge_label.cpu().numpy()
    
    # Compute metrics
    auc = roc_auc_score(edge_label, pred)
    ap = average_precision_score(edge_label, pred)
    
    return auc, ap


@torch.no_grad()
def predict_new_links(model, predictor, data, device, top_k=100):
    """
    Predict new chemical-disease links for drug repositioning
    Returns top-k predictions with highest confidence
    """
    model.eval()
    predictor.eval()
    
    # Get node embeddings
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    num_chemicals = data['chemical'].x.size(0)
    num_diseases = data['disease'].x.size(0)
    
    # Get existing edges to exclude them
    existing_edges = data['chemical', 'chem_disease', 'disease'].edge_index
    existing_edge_set = set(
        zip(existing_edges[0].cpu().numpy(), existing_edges[1].cpu().numpy())
    )
    
    print(f"Computing predictions for drug repositioning...")
    print(f"Total possible combinations: {num_chemicals * num_diseases:,}")
    print(f"Existing edges: {len(existing_edge_set):,}")
    
    # Sample candidate edges (all possible combinations is too large)
    # You can modify this to score all pairs if memory allows
    batch_size = 100000
    all_predictions = []
    
    for chem_start in tqdm(range(0, num_chemicals, 1000)):
        chem_end = min(chem_start + 1000, num_chemicals)
        
        for dis_start in range(0, num_diseases, 1000):
            dis_end = min(dis_start + 1000, num_diseases)
            
            # Create candidate edges
            chem_indices = torch.arange(chem_start, chem_end, device=device)
            dis_indices = torch.arange(dis_start, dis_end, device=device)
            
            # Create all combinations
            chem_grid, dis_grid = torch.meshgrid(chem_indices, dis_indices, indexing='ij')
            edge_index = torch.stack([chem_grid.flatten(), dis_grid.flatten()], dim=0)
            
            # Filter out existing edges
            mask = torch.tensor([
                (c.item(), d.item()) not in existing_edge_set
                for c, d in zip(edge_index[0], edge_index[1])
            ], device=device)
            
            edge_index = edge_index[:, mask]
            
            if edge_index.size(1) == 0:
                continue
            
            # Predict in batches
            for i in range(0, edge_index.size(1), batch_size):
                batch_edge_index = edge_index[:, i:i+batch_size]
                
                scores = predictor(
                    x_dict['chemical'],
                    x_dict['disease'],
                    batch_edge_index
                )
                scores = torch.sigmoid(scores)
                
                # Store predictions
                for j, score in enumerate(scores):
                    chem_idx = batch_edge_index[0, j].item()
                    dis_idx = batch_edge_index[1, j].item()
                    all_predictions.append((chem_idx, dis_idx, score.item()))
    
    # Sort by score and get top-k
    all_predictions.sort(key=lambda x: x[2], reverse=True)
    top_predictions = all_predictions[:top_k]
    
    return top_predictions


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your data (replace this with your actual data loading)
    # Assuming you have already loaded your HeteroData as 'data'
    print("\n=== IMPORTANT ===")
    print("Please load your HeteroData object before running this script.")
    print("Example:")
    print("  data = torch.load('your_heterodata.pt')")
    print("  # Then continue with the code below")
    print("================\n")
    
    # For demonstration, I'll show the complete workflow
    # Uncomment and use your actual data
    
    """
    # Load your data
    data = torch.load('your_heterodata.pt')  # or however you load it
    
    # Print data statistics
    print("Dataset Statistics:")
    print(f"Number of chemicals: {data['chemical'].x.size(0)}")
    print(f"Number of diseases: {data['disease'].x.size(0)}")
    print(f"Number of genes: {data['gene'].x.size(0)}")
    print(f"Chemical-Disease edges: {data['chemical', 'chem_disease', 'disease'].edge_index.size(1)}")
    print(f"Chemical-Gene edges: {data['chemical', 'chem_gene', 'gene'].edge_index.size(1)}")
    print(f"Gene-Disease edges: {data['gene', 'gene_disease', 'disease'].edge_index.size(1)}")
    
    # Split data
    print("\nSplitting data...")
    train_data, val_data, test_data = split_data(
        data, 
        edge_type=('chemical', 'chem_disease', 'disease'),
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Model hyperparameters
    hidden_channels = 128
    out_channels = 64
    num_layers = 3
    
    # Initialize models
    model = HeteroDrugGNN(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        metadata=data.metadata()
    ).to(device)
    
    predictor = LinkPredictor(
        in_channels=out_channels,
        hidden_channels=64,
        out_channels=1,
        num_layers=3
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.001,
        weight_decay=5e-4
    )
    
    # Training
    print("\nStarting training...")
    edge_type = ('chemical', 'chem_disease', 'disease')
    num_epochs = 100
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_epoch(model, predictor, train_data, optimizer, device, edge_type)
        
        # Evaluate
        train_auc, train_ap = evaluate(model, predictor, train_data, device, edge_type)
        val_auc, val_ap = evaluate(model, predictor, val_data, device, edge_type)
        
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
              f'Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}, '
              f'Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save({
                'model': model.state_dict(),
                'predictor': predictor.state_dict(),
            }, 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])
    
    # Final evaluation on test set
    test_auc, test_ap = evaluate(model, predictor, test_data, device, edge_type)
    print(f"\nTest AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Predict new drug-disease associations
    print("\n" + "="*50)
    print("Predicting new drug-disease associations...")
    print("="*50)
    
    top_predictions = predict_new_links(
        model, predictor, data, device, top_k=100
    )
    
    print(f"\nTop 20 predicted drug repositioning candidates:")
    print(f"{'Rank':<6} {'Chemical ID':<12} {'Disease ID':<12} {'Score':<10}")
    print("-" * 50)
    for i, (chem_idx, dis_idx, score) in enumerate(top_predictions[:20], 1):
        print(f"{i:<6} {chem_idx:<12} {dis_idx:<12} {score:.6f}")
    
    # Save all predictions to file
    with open('drug_repositioning_predictions.txt', 'w') as f:
        f.write("Rank,Chemical_ID,Disease_ID,Confidence_Score\n")
        for i, (chem_idx, dis_idx, score) in enumerate(top_predictions, 1):
            f.write(f"{i},{chem_idx},{dis_idx},{score:.6f}\n")
    
    print("\nAll predictions saved to 'drug_repositioning_predictions.txt'")
    """
    
    print("\nWorkflow Summary:")
    print("1. Load your HeteroData")
    print("2. Split into train/val/test")
    print("3. Train the model")
    print("4. Evaluate on test set")
    print("5. Predict new drug-disease associations")
    print("\nUncomment the code in main() and provide your data to run!")


if __name__ == "__main__":
    main()
