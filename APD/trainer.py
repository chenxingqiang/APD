import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


class HybridTrainer:
    def __init__(
        self,
        model,
        config,
        train_dataset,
        val_dataset=None,
        device='cuda',
        learning_rate=2e-4,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Setup dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        ) if val_dataset else None

        # Optimizers
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            dbnet_targets = batch['dbnet_targets'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
            )

            # Calculate losses
            dbnet_loss = outputs.dbnet_output['loss']
            lm_loss = outputs.hidden_states.mean()  # Adjust based on your specific loss

            # Combined loss
            total_batch_loss = dbnet_loss + lm_loss

            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        return total_loss / len(self.train_loader)
