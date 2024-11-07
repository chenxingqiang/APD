# Add these imports at the very top


import torch
from torch.utils.data import DataLoader
import logging
import sys
from pathlib import Path


# Add project root to Python path FIRST, before any APD imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
if True:
    from APD.data import (
        load_dataset,
        IAMDatasetForDBNet,
    )

    from APD.config import APDConfig
    from APD.model import APDModel
    from APD.dbnet import DBNetTrainer


# Import APD components with explicit imports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test():
    # 1. 基础配置
    config = APDConfig()
    config.batch_size = 2  # Smaller batch size
    config.max_epochs = 2  # Just 2 epochs for testing
    config.image_size = (224, 224)  # Add image size configuration

    # 2. 设置设备 - always use CPU
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 3. 加载少量数据
    dataset_path = Path('./datasets/iam_words')
    train_words, validation_words, _ = load_dataset(dataset_path)

    # 4. 创建数据集和加载器 - using all samples since we already have a small dataset
    train_dataset = IAMDatasetForDBNet(train_words, config, is_training=True)
    val_dataset = IAMDatasetForDBNet(
        validation_words, config, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # No multiprocessing for testing
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0  # No multiprocessing for testing
    )

    # 5. 创建模型和训练器
    model = APDModel(config)
    trainer = DBNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # 6. 快速训练测试
    logger.info("Starting quick test training...")
    for epoch in range(config.max_epochs):
        # 训练
        train_loss = trainer.train_one_epoch(epoch)
        logger.info(
            f"Epoch {epoch+1}/{config.max_epochs}, Train Loss: {train_loss:.4f}")

        # 验证
        val_metrics = trainer.validate()
        logger.info(f"Validation Metrics: {val_metrics}")

    logger.info("Quick test completed!")

    # 7. 保存测试结果
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': val_metrics
    }, 'quick_test_model.pth')

    return model, val_metrics


if __name__ == "__main__":
    model, metrics = quick_test()
    print("\nFinal Metrics:", metrics)
