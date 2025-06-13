from my_transformer import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_seq_length=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderOnlyLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt):
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        return nopeak_mask.to(tgt.device)

    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))
        for layer in self.decoder_layers:
            tgt_embedded = layer(tgt_embedded, tgt_mask)  # `encoder_output=None`
        output = self.fc(tgt_embedded)

        return output

    def generate(self, prompt, max_length=100):
        self.eval()
        with torch.no_grad():
            tgt = prompt
            for _ in range(max_length):
                tgt_mask = self.generate_mask(tgt)
                tgt_embedded = self.dropout(
                    self.positional_encoding(self.embedding(tgt))
                )
            for layer in self.decoder_layers:
                tgt_embedded = layer(tgt_embedded, tgt_mask)
                output = self.fc(tgt_embedded[:, -1, :])
                next_token = torch.argmax(
                    F.log_softmax(output, dim=-1), dim=-1
                ).unsqueeze(1)
                if next_token.item() == 102:
                    break  # idk fix this shit bitch
                tgt = torch.cat((tgt, next_token), dim=1)
        return tgt
