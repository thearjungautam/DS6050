from Attention import Attention
import torch
import torch.nn as nn


class DecoderGRU(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.GRU = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.attention = Attention()

        self.linear_layer = nn.Linear(2 * hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, encoder_outputs, src_mask):

        input = input.unsqueeze(1)

        intermediate = self.embedding(input)

        tensor_of_embeddings = self.dropout(intermediate)

        tensor_of_output_features, tensor_with_final_hidden_states = self.GRU(
            tensor_of_embeddings, hidden
        )

        tensor_of_output_features = tensor_of_output_features.squeeze(1)

        context_matrix, matrix_of_attention_weights = self.attention(
            tensor_of_output_features, encoder_outputs, src_mask
        )

        tensor_of_output_features_and_context = torch.cat(
            (tensor_of_output_features, context_matrix), dim=1
        )

        tensor_of_logits = self.linear_layer(tensor_of_output_features_and_context)

        return tensor_of_logits, tensor_with_final_hidden_states, matrix_of_attention_weights