# Drug Repositioning using Heterogeneous Graph Neural Networks

Complete implementation for predicting new drug-disease associations using link prediction on heterogeneous biomedical graphs.

## 📋 Overview

This implementation uses your HeteroData with:
- **Chemical** nodes (drugs/compounds)
- **Disease** nodes 
- **Gene** nodes
- **Chemical-Gene**, **Chemical-Disease**, and **Gene-Disease** edges

The goal is to predict new **Chemical-Disease** associations for drug repositioning.

## 🚀 Quick Start

### 1. Prerequisites

```bash
pip install torch torch-geometric numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 2. Load Your Data

```python
import torch

# Load your HeteroData
data = torch.load('your_heterodata.pt')
# or
# data = your_loading_function()
```

### 3. Run the Pipeline

```python
from run_pipeline import run_pipeline

# Run complete pipeline
model, predictor, predictions = run_pipeline(
    data, 
    num_epochs=100,
    hidden_channels=128,
    out_channels=64
)
```

### 4. Get Predictions

The pipeline automatically generates:
- `drug_repositioning_predictions.txt` - Top 1000 predictions
- `filtered_predictions.txt` - Predictions with metapath support
- `repositioning_report.txt` - Detailed analysis report
- `best_model.pt` - Trained model checkpoint

## 📁 File Descriptions

### Core Files

1. **`drug_repositioning_gnn.py`** - Main GNN implementation
   - `HeteroDrugGNN`: Heterogeneous graph neural network
   - `LinkPredictor`: Link prediction head
   - Training and evaluation functions
   - Prediction generation

2. **`drug_repositioning_utils.py`** - Analysis utilities
   - Data analysis functions
   - Visualization tools
   - Metapath-based filtering
   - Report generation

3. **`run_pipeline.py`** - End-to-end pipeline
   - Complete workflow from training to predictions
   - Automated analysis and reporting

## 🔧 Detailed Usage

### Option A: Use the Complete Pipeline (Recommended)

```python
from run_pipeline import run_pipeline

# Simple usage
model, predictor, predictions = run_pipeline(data)

# With custom parameters
model, predictor, predictions = run_pipeline(
    data,
    num_epochs=150,
    hidden_channels=256,
    out_channels=128
)
```

### Option B: Step-by-Step Manual Approach

```python
import torch
from drug_repositioning_gnn import (
    HeteroDrugGNN, LinkPredictor, split_data, 
    train_epoch, evaluate, predict_new_links
)
from drug_repositioning_utils import analyze_heterodata

# 1. Analyze data
analyze_heterodata(data)

# 2. Split data
train_data, val_data, test_data = split_data(data)

# 3. Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroDrugGNN(
    hidden_channels=128,
    out_channels=64,
    num_layers=3,
    metadata=data.metadata()
).to(device)

predictor = LinkPredictor(
    in_channels=64,
    hidden_channels=64
).to(device)

# 4. Train
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(predictor.parameters()),
    lr=0.001
)

train_data = train_data.to(device)
val_data = val_data.to(device)

for epoch in range(100):
    loss = train_epoch(
        model, predictor, train_data, optimizer, 
        device, ('chemical', 'chem_disease', 'disease')
    )
    val_auc, val_ap = evaluate(
        model, predictor, val_data, 
        device, ('chemical', 'chem_disease', 'disease')
    )
    print(f'Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

# 5. Predict new associations
predictions = predict_new_links(model, predictor, data, device, top_k=100)

# 6. Analyze
from drug_repositioning_utils import analyze_predictions, create_prediction_report
analyze_predictions(predictions, data)
create_prediction_report(predictions, data)
```

## 🎯 Model Architecture

### HeteroDrugGNN
- Uses heterogeneous message passing (HeteroConv)
- Separate parameters for each edge type
- 3-layer architecture with ReLU activations
- GraphSAGE convolutions for efficient neighborhood aggregation

### LinkPredictor
- MLP-based prediction head
- Hadamard product of node embeddings
- Binary classification (link exists or not)

## 📊 Understanding the Output

### Prediction Files

**drug_repositioning_predictions.txt**
```
Rank,Chemical_ID,Disease_ID,Confidence_Score
1,1234,567,0.987654
2,2345,678,0.976543
...
```

**filtered_predictions.txt** (with metapath support)
```
Rank,Chemical_ID,Disease_ID,Score,Common_Genes
1,1234,567,0.987654,5
2,2345,678,0.976543,3
...
```

### Interpreting Scores

- **Score range**: 0 to 1 (sigmoid output)
- **Higher scores** = stronger predicted association
- **Threshold**: Typically 0.5, but depends on your use case
- **Common genes**: Number of genes that link chemical to disease (metapath evidence)

## 🔍 Advanced Analysis

### Filter by Metapath Support

```python
from drug_repositioning_utils import filter_predictions_by_metapath

# Only keep predictions with at least 2 common genes
filtered = filter_predictions_by_metapath(
    predictions, 
    data, 
    min_common_genes=2
)
```

### Investigate Specific Predictions

```python
from drug_repositioning_utils import get_metapath_features

# Check why a specific chemical-disease pair was predicted
features = get_metapath_features(data, chemical_idx=100, disease_idx=50)
print(f"Common genes: {features['num_common_genes']}")
print(f"Chemical degree: {features['chemical_degree']}")
print(f"Disease degree: {features['disease_degree']}")
```

### Custom Visualization

```python
from drug_repositioning_utils import plot_degree_distribution, plot_prediction_scores

# Plot degree distributions
plot_degree_distribution(data, ('chemical', 'chem_disease', 'disease'))

# Plot prediction scores
plot_prediction_scores(predictions)
```

## ⚙️ Hyperparameter Tuning

Key hyperparameters to tune:

```python
# Model architecture
hidden_channels = 128      # Size of hidden layers (64, 128, 256)
out_channels = 64          # Embedding dimension (32, 64, 128)
num_layers = 3             # Number of GNN layers (2, 3, 4)

# Training
learning_rate = 0.001      # Learning rate (0.0001 - 0.01)
weight_decay = 5e-4        # L2 regularization (0, 1e-5, 1e-4)
num_epochs = 100           # Training epochs (50 - 200)
patience = 10              # Early stopping patience

# Data split
val_ratio = 0.1            # Validation set ratio
test_ratio = 0.1           # Test set ratio
```

## 🎓 Methodology

### 1. Message Passing
The model aggregates information from:
- Chemical → Gene → Disease paths
- Direct Chemical → Disease connections
- Gene → Disease associations

### 2. Link Prediction
For a chemical-disease pair (c, d):
1. Get embeddings: h_c and h_d
2. Combine: z = h_c ⊙ h_d (Hadamard product)
3. MLP: score = MLP(z)
4. Sigmoid: p = σ(score)

### 3. Training Objective
Binary cross-entropy loss with:
- Positive samples: existing chemical-disease edges
- Negative samples: randomly sampled non-existing edges (1:1 ratio)

## 📈 Evaluation Metrics

- **AUC-ROC**: Area under ROC curve (overall discrimination)
- **Average Precision (AP)**: Precision-recall area (handles class imbalance)
- **Hits@K**: Fraction of true positives in top K predictions

## 💡 Tips for Better Results

1. **Feature Engineering**
   - Use chemical fingerprints (ECFP, MACCS)
   - Use disease embeddings (text descriptions)
   - Use gene expression data

2. **Data Balancing**
   - Your chemical-disease edges dominate (2.9M vs 34K gene-disease)
   - Consider edge reweighting or sampling strategies

3. **Negative Sampling**
   - Use hard negatives (chemical-disease pairs with intermediate similarity)
   - Avoid false negatives (actual associations not in your data)

4. **Ensemble Methods**
   - Train multiple models with different seeds
   - Average predictions for more robust results

5. **External Validation**
   - Check top predictions against DrugBank, CTD, or literature
   - Validate with experimental data if available

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in predict_new_links
# Edit drug_repositioning_gnn.py line ~215:
batch_size = 50000  # Default is 100000
```

### Slow Training
```python
# Use fewer layers or smaller hidden dimensions
model = HeteroDrugGNN(
    hidden_channels=64,   # Instead of 128
    out_channels=32,      # Instead of 64
    num_layers=2          # Instead of 3
)
```

### Poor Performance
- Check data quality and splits
- Increase model capacity
- Train for more epochs
- Adjust learning rate
- Add features to node representations

## 📚 References

1. **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
2. **Heterogeneous GNN**: Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks"
3. **Drug Repositioning**: Zeng et al., "deepDR: a network-based deep learning approach to in silico drug repositioning"

## 📝 Citation

If you use this code, please cite:
```
@software{drug_repositioning_gnn,
  title={Heterogeneous GNN for Drug Repositioning},
  year={2026},
  url={https://github.com/yourusername/drug-repositioning}
}
```

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Share results

## 📄 License

MIT License - feel free to use and modify for your research!

---

**Good luck with your drug repositioning project! 🚀💊**
