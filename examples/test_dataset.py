from APD.data import load_dataset
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_dataset():
    dataset_path = Path('./datasets/iam_words')

    # Load small test dataset
    train_words, val_words, test_words = load_dataset(dataset_path)

    # Print some information about the dataset
    print("\nDataset samples:")
    for word in train_words[:2]:  # Print first 2 training samples
        print(f"ID: {word.id}")
        print(f"File: {word.file_path}")
        print(f"Writer: {word.writer_id}")
        print(f"Text: {word.transcription}")
        print("---")


if __name__ == "__main__":
    test_dataset()
