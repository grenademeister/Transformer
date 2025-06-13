from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    dataset = load_dataset(
        "wikipedia", "20220301.simple", split="train", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk("./tokenized_wiki")


class WikiDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]["input_ids"])
