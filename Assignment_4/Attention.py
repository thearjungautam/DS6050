import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def forward(self, decoder_hidden, encoder_outputs, src_mask = None):

        tensor_of_current_decoder_hidden_states = decoder_hidden.unsqueeze(2)

        tensor_of_alignment_scores = torch.bmm(
            encoder_outputs, tensor_of_current_decoder_hidden_states
        )
        
        matrix_of_alignment_scores = tensor_of_alignment_scores.squeeze(2)

        if src_mask is not None:
            matrix_of_alignment_scores = matrix_of_alignment_scores.masked_fill(
                ~src_mask, -1e9
            )
        
        matrix_of_attention_weights = F.softmax(matrix_of_alignment_scores, dim=1)

        tensor_of_attention_weights = matrix_of_attention_weights.unsqueeze(1)

        context_tensor = torch.bmm(
            tensor_of_attention_weights, encoder_outputs
        )
        
        context_matrix = context_tensor.squeeze(1)

        return context_matrix, matrix_of_attention_weights