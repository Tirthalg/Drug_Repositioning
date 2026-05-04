"""
Example Script: End-to-End Drug Repositioning Pipeline
This shows how to use all the components together with your HeteroData
"""

import torch
from drug_repositioning_gnn import (
    HeteroDrugGNN, 
    LinkPredictor, 
    split_data, 
    train_epoch, 
    evaluate,
    predict_new_links
)
from drug_repositioning_utils import (
    analyze_heterodata,
    plot_degree_distribution,
    analyze_predictions,
    plot_prediction_scores,
    create_prediction_report,
    filter_predictions_by_metapath
)


def run_pipeline(data, num_epochs=100, hidden_channels=128, out_channels=64):
    """
    Complete pipeline for drug repositioning
    
    Args:
        data: Your HeteroData object
        num_epochs: Number of training epochs
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Step 1: Analyze the data
    print("STEP 1: Data Analysis")
    print("-" * 60)
    analyze_heterodata(data)
    plot_degree_distribution(data, ('chemical', 'chem_disease', 'disease'))
    
    # Step 2: Split the data
    print("\nSTEP 2: Data Splitting")
    print("-" * 60)
    train_data, val_data, test_data = split_data(
        data, 
        edge_type=('chemical', 'chem_disease', 'disease'),
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print(f"Train edges: {train_data['chemical', 'chem_disease', 'disease'].edge_label_index.size(1)}")
    print(f"Val edges: {val_data['chemical', 'chem_disease', 'disease'].edge_label_index.size(1)}")
    print(f"Test edges: {test_data['chemical', 'chem_disease', 'disease'].edge_label_index.size(1)}")
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Step 3: Initialize models
    print("\nSTEP 3: Model Initialization")
    print("-" * 60)
    model = HeteroDrugGNN(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=3,
        metadata=data.metadata()
    ).to(device)
    
    predictor = LinkPredictor(
        in_channels=out_channels,
        hidden_channels=64,
        out_channels=1,
        num_layers=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in predictor.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Step 4: Train the model
    print("\nSTEP 4: Model Training")
    print("-" * 60)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.001,
        weight_decay=5e-4
    )
    
    edge_type = ('chemical', 'chem_disease', 'disease')
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_epoch(model, predictor, train_data, optimizer, device, edge_type)
        
        # Evaluate
        train_auc, train_ap = evaluate(model, predictor, train_data, device, edge_type)
        val_auc, val_ap = evaluate(model, predictor, val_data, device, edge_type)
        
        if epoch % 5 == 0 or epoch == 1:
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
                'epoch': epoch,
                'val_auc': val_auc,
            }, 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Step 5: Load best model and evaluate
    print("\nSTEP 5: Final Evaluation")
    print("-" * 60)
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_auc, test_ap = evaluate(model, predictor, test_data, device, edge_type)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    
    # Step 6: Predict new drug-disease associations
    print("\nSTEP 6: Drug Repositioning Predictions")
    print("-" * 60)
    
    # Move data back to device for prediction
    data = data.to(device)
    
    top_predictions = predict_new_links(
        model, predictor, data, device, top_k=1000
    )
    
    # Step 7: Analyze predictions
    print("\nSTEP 7: Prediction Analysis")
    print("-" * 60)
    
    # Basic analysis
    df = analyze_predictions(top_predictions, data)
    
    # Plot scores
    plot_prediction_scores(top_predictions)
    
    # Create detailed report
    create_prediction_report(top_predictions, data)
    
    # Filter by metapath support
    print("\nFiltering predictions by metapath support...")
    filtered_predictions = filter_predictions_by_metapath(
        top_predictions[:100], 
        data.cpu(), 
        min_common_genes=1
    )
    
    print("\nTop 10 predictions with metapath support:")
    print(f"{'Rank':<6} {'Chemical':<12} {'Disease':<12} {'Score':<10} {'Common Genes':<15}")
    print("-" * 60)
    for i, (chem_idx, dis_idx, score, num_genes) in enumerate(filtered_predictions[:10], 1):
        print(f"{i:<6} {chem_idx:<12} {dis_idx:<12} {score:<10.6f} {num_genes:<15}")
    
    # Save filtered predictions
    with open('filtered_predictions.txt', 'w') as f:
        f.write("Rank,Chemical_ID,Disease_ID,Score,Common_Genes\n")
        for i, (chem_idx, dis_idx, score, num_genes) in enumerate(filtered_predictions, 1):
            f.write(f"{i},{chem_idx},{dis_idx},{score:.6f},{num_genes}\n")
    
    print("\nFiltered predictions saved to 'filtered_predictions.txt'")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - best_model.pt (trained model)")
    print("  - drug_repositioning_predictions.txt (all predictions)")
    print("  - filtered_predictions.txt (predictions with metapath support)")
    print("  - repositioning_report.txt (detailed report)")
    print("  - degree_distribution.png (visualization)")
    print("  - prediction_scores.png (visualization)")
    
    return model, predictor, top_predictions


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("DRUG REPOSITIONING PIPELINE")
    print("="*60)
    print("\nThis script demonstrates the complete workflow.")
    print("To run with your data:")
    print("\n1. Load your HeteroData:")
    print("   data = torch.load('your_data.pt')")
    print("   # or however you load your data")
    print("\n2. Run the pipeline:")
    print("   model, predictor, predictions = run_pipeline(data)")
    print("\n3. Use the predictions for drug repositioning!")
    print("="*60)
    
    # Example of how to use it:
    """
    # Load your data
    data = torch.load('your_heterodata.pt')
    
    # Run the complete pipeline
    model, predictor, predictions = run_pipeline(
        data, 
        num_epochs=100,
        hidden_channels=128,
        out_channels=64
    )
    
    # The predictions are now ready for further analysis!
    """
