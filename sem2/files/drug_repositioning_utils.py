"""
Utility functions for drug repositioning analysis
Includes data exploration, visualization, and result interpretation
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_heterodata(data):
    """
    Analyze and print statistics about the heterogeneous graph
    """
    print("="*60)
    print("HETEROGENEOUS GRAPH ANALYSIS")
    print("="*60)
    
    # Node statistics
    print("\n--- Node Statistics ---")
    for node_type in data.node_types:
        num_nodes = data[node_type].x.size(0)
        num_features = data[node_type].x.size(1)
        print(f"{node_type.capitalize()}:")
        print(f"  Number of nodes: {num_nodes:,}")
        print(f"  Feature dimension: {num_features}")
    
    # Edge statistics
    print("\n--- Edge Statistics ---")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.size(1)
        src_type, rel_type, dst_type = edge_type
        print(f"{src_type} --[{rel_type}]--> {dst_type}:")
        print(f"  Number of edges: {num_edges:,}")
        
        # Calculate sparsity
        src_nodes = data[src_type].x.size(0)
        dst_nodes = data[dst_type].x.size(0)
        possible_edges = src_nodes * dst_nodes
        sparsity = (num_edges / possible_edges) * 100
        print(f"  Sparsity: {sparsity:.4f}%")
        print()
    
    # Degree statistics
    print("--- Degree Statistics ---")
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        src_type, rel_type, dst_type = edge_type
        
        # Source node degrees (out-degree)
        src_degrees = torch.bincount(edge_index[0])
        print(f"{src_type} out-degrees ({rel_type}):")
        print(f"  Mean: {src_degrees.float().mean():.2f}")
        print(f"  Median: {src_degrees.float().median():.2f}")
        print(f"  Max: {src_degrees.max()}")
        print(f"  Min: {src_degrees.min()}")
        
        # Target node degrees (in-degree)
        dst_degrees = torch.bincount(edge_index[1])
        print(f"{dst_type} in-degrees ({rel_type}):")
        print(f"  Mean: {dst_degrees.float().mean():.2f}")
        print(f"  Median: {dst_degrees.float().median():.2f}")
        print(f"  Max: {dst_degrees.max()}")
        print(f"  Min: {dst_degrees.min()}")
        print()


def plot_degree_distribution(data, edge_type, save_path='degree_distribution.png'):
    """
    Plot degree distribution for a specific edge type
    """
    edge_index = data[edge_type].edge_index
    src_type, rel_type, dst_type = edge_type
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Source degrees
    src_degrees = torch.bincount(edge_index[0]).numpy()
    axes[0].hist(src_degrees, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{src_type.capitalize()} Out-Degree Distribution')
    axes[0].set_yscale('log')
    
    # Target degrees
    dst_degrees = torch.bincount(edge_index[1]).numpy()
    axes[1].hist(dst_degrees, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{dst_type.capitalize()} In-Degree Distribution')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Degree distribution plot saved to {save_path}")
    plt.close()


def analyze_predictions(predictions, data, chemical_names=None, disease_names=None):
    """
    Analyze and interpret predictions
    
    Args:
        predictions: List of (chemical_idx, disease_idx, score) tuples
        data: HeteroData object
        chemical_names: Optional dict mapping chemical indices to names
        disease_names: Optional dict mapping disease indices to names
    """
    print("="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions, columns=['Chemical_ID', 'Disease_ID', 'Score'])
    
    print(f"\nTotal predictions: {len(df)}")
    print(f"Score range: [{df['Score'].min():.4f}, {df['Score'].max():.4f}]")
    print(f"Mean score: {df['Score'].mean():.4f}")
    print(f"Median score: {df['Score'].median():.4f}")
    
    # Score distribution
    print("\nScore distribution:")
    print(df['Score'].describe())
    
    # Most frequently predicted chemicals and diseases
    print("\n--- Most Frequently Predicted Chemicals (Top 10) ---")
    top_chemicals = df['Chemical_ID'].value_counts().head(10)
    for chem_id, count in top_chemicals.items():
        name = chemical_names.get(chem_id, f"Chemical_{chem_id}") if chemical_names else f"Chemical_{chem_id}"
        print(f"{name}: {count} predictions")
    
    print("\n--- Most Frequently Predicted Diseases (Top 10) ---")
    top_diseases = df['Disease_ID'].value_counts().head(10)
    for dis_id, count in top_diseases.items():
        name = disease_names.get(dis_id, f"Disease_{dis_id}") if disease_names else f"Disease_{dis_id}"
        print(f"{name}: {count} predictions")
    
    return df


def plot_prediction_scores(predictions, save_path='prediction_scores.png'):
    """
    Plot prediction score distribution
    """
    scores = [p[2] for p in predictions]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prediction Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Prediction Scores')
    axes[0].axvline(np.median(scores), color='red', linestyle='--', label=f'Median: {np.median(scores):.4f}')
    axes[0].legend()
    
    # Cumulative distribution
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cumulative)
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution of Prediction Scores')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction score plot saved to {save_path}")
    plt.close()


def create_prediction_report(predictions, data, output_file='repositioning_report.txt',
                            chemical_names=None, disease_names=None):
    """
    Create a detailed report of predictions
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DRUG REPOSITIONING PREDICTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total predictions: {len(predictions)}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 100 PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Rank':<8}{'Chemical ID':<15}{'Disease ID':<15}{'Score':<12}{'Chemical Name':<30}{'Disease Name':<30}\n")
        f.write("-"*80 + "\n")
        
        for i, (chem_idx, dis_idx, score) in enumerate(predictions[:100], 1):
            chem_name = chemical_names.get(chem_idx, 'Unknown') if chemical_names else 'N/A'
            dis_name = disease_names.get(dis_idx, 'Unknown') if disease_names else 'N/A'
            
            f.write(f"{i:<8}{chem_idx:<15}{dis_idx:<15}{score:<12.6f}{chem_name:<30}{dis_name:<30}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        scores = [p[2] for p in predictions]
        f.write(f"Score Statistics:\n")
        f.write(f"  Mean: {np.mean(scores):.6f}\n")
        f.write(f"  Median: {np.median(scores):.6f}\n")
        f.write(f"  Std: {np.std(scores):.6f}\n")
        f.write(f"  Min: {np.min(scores):.6f}\n")
        f.write(f"  Max: {np.max(scores):.6f}\n")
    
    print(f"Detailed report saved to {output_file}")


def get_metapath_features(data, chemical_idx, disease_idx):
    """
    Extract metapath-based features for a chemical-disease pair
    Useful for understanding why a prediction was made
    """
    features = {}
    
    # Get edges
    chem_gene_edges = data['chemical', 'chem_gene', 'gene'].edge_index
    gene_disease_edges = data['gene', 'gene_disease', 'disease'].edge_index
    
    # Find genes connected to the chemical
    genes_from_chem = chem_gene_edges[1][chem_gene_edges[0] == chemical_idx].unique()
    
    # Find genes connected to the disease
    genes_to_disease = gene_disease_edges[0][gene_disease_edges[1] == disease_idx].unique()
    
    # Find common genes (metapath: chemical -> gene -> disease)
    common_genes = torch.tensor(list(set(genes_from_chem.tolist()) & set(genes_to_disease.tolist())))
    
    features['num_common_genes'] = len(common_genes)
    features['chemical_degree'] = (chem_gene_edges[0] == chemical_idx).sum().item()
    features['disease_degree'] = (gene_disease_edges[1] == disease_idx).sum().item()
    
    return features


def filter_predictions_by_metapath(predictions, data, min_common_genes=1):
    """
    Filter predictions to only include those with supporting metapath evidence
    """
    filtered = []
    
    print(f"Filtering {len(predictions)} predictions by metapath support...")
    
    for chem_idx, dis_idx, score in predictions:
        features = get_metapath_features(data, chem_idx, dis_idx)
        if features['num_common_genes'] >= min_common_genes:
            filtered.append((chem_idx, dis_idx, score, features['num_common_genes']))
    
    print(f"Retained {len(filtered)} predictions with at least {min_common_genes} common gene(s)")
    
    return filtered


# Example usage function
def example_usage():
    """
    Example of how to use these utility functions
    """
    print("""
    Example Usage:
    
    # 1. Analyze your heterogeneous data
    analyze_heterodata(data)
    
    # 2. Plot degree distributions
    plot_degree_distribution(data, ('chemical', 'chem_disease', 'disease'))
    
    # 3. After getting predictions from the model
    df = analyze_predictions(predictions, data)
    
    # 4. Plot prediction scores
    plot_prediction_scores(predictions)
    
    # 5. Create detailed report
    create_prediction_report(predictions, data)
    
    # 6. Filter by metapath support
    filtered_preds = filter_predictions_by_metapath(predictions, data, min_common_genes=2)
    
    # 7. Investigate specific prediction
    chem_idx, dis_idx = 100, 50
    features = get_metapath_features(data, chem_idx, dis_idx)
    print(f"Metapath features: {features}")
    """)


if __name__ == "__main__":
    example_usage()
