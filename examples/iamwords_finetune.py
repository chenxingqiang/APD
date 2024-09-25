



import pickle
import os
import sys
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.modeling_gpt2 import GPT2Config
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

if True:
    from APD.config import APDConfig
    from APD.processor import APDProcessor
    from APD.model import APDLMHeadModel

@dataclass
class Word:
    id: str
    file_path: Path
    writer_id: str
    transcription: str


def get_words_from_xml(xml_file: str, word_image_files: List[str]) -> List[Word]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    root_id = root.get('id')
    writer_id = root.get('writer-id')
    xml_words = []
    for line in root.findall('handwritten-part')[0].findall('line'):
        for word in line.findall('word'):
            image_file = Path(
                [f for f in word_image_files if f.endswith(word.get('id') + '.png')][0])
            try:
                with Image.open(image_file) as _:
                    xml_words.append(
                        Word(
                            id=root_id,
                            file_path=image_file,
                            writer_id=writer_id,
                            transcription=word.get('text')
                        )
                    )
            except Exception:
                pass
    return xml_words


class IAMDataset(Dataset):
    def __init__(self, words: List[Word], config: APDConfig):
        self.words = words
        self.processor = APDProcessor(
            config, add_eos_token=True, add_bos_token=True)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        inputs = self.processor(
            images=Image.open(self.words[item].file_path).convert('RGB'),
            texts=self.words[item].transcription,
            padding='max_length',
            return_tensors="pt",
            return_labels=True,
        )
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.labels[0]
        }


def save_dataset(dataset_path: Path, train_words: List[Word], validation_words: List[Word], test_words: List[Word]):
    data = {
        'train_words': train_words,
        'validation_words': validation_words,
        'test_words': test_words
    }
    with open(dataset_path / 'processed_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {dataset_path / 'processed_dataset.pkl'}")


def load_dataset(dataset_path: Path) -> Tuple[List[Word], List[Word], List[Word]]:
    processed_file = dataset_path / 'processed_dataset.pkl'
    if os.path.exists(processed_file):
        print(f"Loading processed dataset from {processed_file}")
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        train_words = data['train_words']
        validation_words = data['validation_words']
        test_words = data['test_words']
        print(
            f'Loaded dataset - Train size: {len(train_words)}; Validation size: {len(validation_words)}; Test size: {len(test_words)}')
    else:
        print("Processed dataset not found. Generating new dataset.")
        train_words, validation_words, test_words = generate_dataset(
            dataset_path)
        save_dataset(dataset_path, train_words, validation_words, test_words)

    return train_words, validation_words, test_words


def generate_dataset(dataset_path: Path) -> Tuple[List[Word], List[Word], List[Word]]:
    xml_files = sorted(glob.glob(str(dataset_path / 'xml' / '*.xml')))
    word_image_files = sorted(
        glob.glob(str(dataset_path / 'words' / '**' / '*.png'), recursive=True))
    print(f"{len(xml_files)} XML files and {len(word_image_files)} word image files")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        words_from_xmls = list(
            tqdm.tqdm(
                pool.imap(partial(get_words_from_xml,
                          word_image_files=word_image_files), xml_files),
                total=len(xml_files),
                desc='Building dataset'
            )
        )

    words = [word for words in words_from_xmls for word in words]

    # Load train/validation/test splits
    with open(dataset_path / 'splits' / 'train.uttlist') as fp:
        train_ids = set(line.strip() for line in fp)
    with open(dataset_path / 'splits' / 'test.uttlist') as fp:
        test_ids = set(line.strip() for line in fp)
    with open(dataset_path / 'splits' / 'validation.uttlist') as fp:
        validation_ids = set(line.strip() for line in fp)

    train_words = [word for word in words if word.id in train_ids]
    validation_words = [word for word in words if word.id in validation_ids]
    test_words = [word for word in words if word.id in test_ids]

    print(
        f'Generated dataset - Train size: {len(train_words)}; Validation size: {len(validation_words)}; Test size: {len(test_words)}')
    return train_words, validation_words, test_words

def create_dataloaders(train_words: List[Word], validation_words: List[Word], test_words: List[Word], config: APDConfig, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = IAMDataset(words=train_words, config=config)
    validation_data = IAMDataset(words=validation_words, config=config)
    test_data = IAMDataset(words=test_words, config=config)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count())
    validation_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count())
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count())

    return train_dataloader, validation_dataloader, test_dataloader



def train_model(model: torch.nn.Module, train_dataloader: DataLoader, validation_dataloader: DataLoader, epochs: int = 50, lr: float = 1e-4, device: str = 'mps'):
    model.to(device)
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_losses, epoch_accuracies = [], []
        for inputs in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimiser.zero_grad()
            outputs = model(**inputs)

            outputs.loss.backward()
            optimiser.step()

            epoch_losses.append(outputs.loss.item())
            epoch_accuracies.append(outputs.accuracy.item())

        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))

        validation_loss, validation_accuracy = evaluate_model(
            model, validation_dataloader, device)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch + 1} - Train loss: {train_losses[-1]:.4f}, Train accuracy: {train_accuracies[-1]:.4f}, "
              f"Validation loss: {validation_losses[-1]:.4f}, Validation accuracy: {validation_accuracies[-1]:.4f}")

    return model, train_losses, train_accuracies, validation_losses, validation_accuracies


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str = 'mps') -> Tuple[float, float]:
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            losses.append(outputs.loss.item())
            accuracies.append(outputs.accuracy.item())

    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def test_model(model: torch.nn.Module, test_words: List[Word], processor: APDProcessor, num_samples: int = 50):
    model.eval()
    model.to('cpu')

    for test_word in test_words[:num_samples]:
        image = Image.open(test_word.file_path).convert('RGB')
        inputs = processor(
            images=image,
            texts=processor.tokeniser.bos_token,
            return_tensors='pt'
        )
        model_output = model.generate(inputs, processor, num_beams=3)

        predicted_text = processor.tokeniser.decode(
            model_output[0], skip_special_tokens=True)

        plt.figure(figsize=(10, 5))
        plt.title(predicted_text, fontsize=24)
        plt.imshow(np.array(image, dtype=np.uint8))
        plt.xticks([]), plt.yticks([])
        plt.show()


def main():
    dataset_path = Path('./datasets/iam_words')
    config = APDConfig(gpt2_hf_model='openai-community/gpt2')

    train_words, validation_words, test_words = load_dataset(dataset_path)
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(
         train_words, validation_words, test_words, config)

    print("Pre-trained GPT2 config:",
         GPT2Config.from_pretrained('openai-community/gpt2'))


    print("Custom model config:", config.gpt2_config)

    model = APDLMHeadModel(config)

    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")

    trained_model, train_losses, train_accuracies, validation_losses, validation_accuracies = train_model(
        model, train_dataloader, validation_dataloader, device=device)

    # # Plot training results
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.legend()
    # plt.title('Loss')
    # plt.subplot(1, 2, 2)
    # plt.plot(train_accuracies, label='Train Accuracy')
    # plt.plot(validation_accuracies, label='Validation Accuracy')
    # plt.legend()
    # plt.title('Accuracy')
    # plt.show()

    # # Test the model
    # processor = APDProcessor(config)
    # test_model(trained_model, test_words, processor)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
