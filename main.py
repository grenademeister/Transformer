import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

from tokenizer import WikiDataset
from decoder_transformer import DecoderOnlyTransformer

if __name__ == "__main__":
    tokenized_datasets = load_from_disk("./tokenized_wiki")
    wiki = WikiDataset(tokenized_datasets)

    # 6. PyTorch Dataset 및 DataLoader 준비
    train_dataloader = DataLoader(wiki, batch_size=64, shuffle=True)

    # 7. 모델 및 학습 설정
    vocab_size = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    ).vocab_size  # 실제 토큰 개수
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 1024
    max_seq_length = 128
    dropout = 0.1

    model = DecoderOnlyTransformer(
        vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 8. 학습 루프
    for epoch in range(10):
        for batch in train_dataloader:
            optimizer.zero_grad()

            tgt = batch  # Decoder-Only 모델이므로 `tgt`만 입력
            output = model(tgt[:, :-1])  # 마지막 토큰 제외
            loss = criterion(
                output.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1)
            )

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
