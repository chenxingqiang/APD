import sys
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from PIL import Image
import multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to Python path FIRST, before any APD imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
if True:
    from APD import (
        APDConfig,
        load_dataset,
        IAMDatasetForDBNet,
        APDModel,
        DBNetTrainer,
        DBNetInference,
        Visualizer
    )


def main():
    dataset_path = Path('./datasets/iam_words')
    config = APDConfig()

    # Use original training parameters
    config.max_epochs = 10
    config.batch_size = 32
    config.num_workers = 4

    # Load the full dataset
    train_words, validation_words, test_words = load_dataset(dataset_path)
    print(f"Training with full dataset - Train: {len(train_words)}, "
          f"Val: {len(validation_words)}, Test: {len(test_words)}")

    # Create datasets
    train_dataset = IAMDatasetForDBNet(train_words, config, is_training=True)
    val_dataset = IAMDatasetForDBNet(
        validation_words, config, is_training=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # Create model
    model = APDModel(config)
    device = torch.device("cpu")  # Force CPU for testing

    # Initialize trainer with loaders instead of datasets
    trainer = DBNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,  # Changed from train_dataset
        val_loader=val_loader,      # Changed from val_dataset
        device=device
    )

    # Add tensorboard logging
    writer = SummaryWriter('runs/dbnet_test')

    # Training loop with enhanced logging
    best_hmean = 0
    for epoch in range(config.max_epochs):
        # Train
        train_loss = trainer.train_one_epoch(epoch)

        # Validate
        val_metrics = trainer.validate()

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
        writer.add_scalar('Metrics/hmean', val_metrics['hmean'], epoch)

        print(f"Epoch {epoch+1}/{config.max_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")
        print("---")

        # Save best model
        if val_metrics['hmean'] > best_hmean:
            best_hmean = val_metrics['hmean']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_hmean': best_hmean,
            }, 'best_model.pth')

    writer.close()

    # Test inference with visualization
    inferencer = DBNetInference(model=model, config=config, device=device)
    visualizer = Visualizer()

    # Test inference with visualization
    output_dir = Path('output_visualizations')
    output_dir.mkdir(exist_ok=True)

    # Only test first sample
    test_word = test_words[0]
    image = Image.open(test_word.file_path).convert('RGB')
    image_np = np.array(image)

    # Detect text regions
    detected_regions = inferencer.detect(image_np)

    # Visualize and save results
    save_path = output_dir / 'detection_test.png'
    visualizer.visualize_detection(
        image_np,
        detected_regions,
        save_path=str(save_path)
    )

    print(f"Original text: {test_word.transcription}")
    print(f"Detected regions: {len(detected_regions)}")
    print(f"Visualization saved to {save_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
