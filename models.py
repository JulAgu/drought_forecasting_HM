import torch
import torch.nn as nn
import torch.nn.functional as F

class DroughtNetLSTM(nn.Module):
    """
    The proposed LSTM Baseline for drought prediction using the DroughtED dataset.
    """
    def __init__(
        self,
        output_size,
        num_input_features,
        hidden_dim,
        n_layers,
        ffnn_layers,
        drop_prob,
        static_dim,
    ):
        """
        Initializes the LSTM Baseline.

        Parameters
        ----------
        output_size : int
            The size of the output vector.
        num_input_features : int
            The number of variables in the time series.
        hidden_dim : int
            The size of the LSTM hidden state.
        n_layers : int
            The number of layers in the LSTM.
        ffnn_layers : int
            The number of layers in the FFNN processing the combined data.
        drop_prob : float
            The dropout probability for the LSTM.
        static_dim : int
            The size of the static data.
        """
        super(DroughtNetLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            num_input_features,
            hidden_dim,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fflayers = []
        for i in range(ffnn_layers - 1):
            if i == 0:
                self.fflayers.append(nn.Linear(hidden_dim + static_dim, hidden_dim))
            else:
                self.fflayers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fflayers = nn.ModuleList(self.fflayers)
        self.final = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden, static=None):
        batch_size = x.size(0)
        x = x.to(dtype=torch.float32)
        if static is not None:
            static = static.to(dtype=torch.float32)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]

        out = self.dropout(lstm_out)
        for i in range(len(self.fflayers)):
            if i == 0 and static is not None:
                out = self.fflayers[i](torch.cat((out, static), 1))
            else:
                out = self.fflayers[i](out)
        out = self.final(out)

        out = out.view(batch_size, -1)
        return out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )
        return hidden

class HybridModel(nn.Module):
    """
    The proposed hybrid model (HM) That combines:
    - LSTMs.
    - Attention Mechanism.
    - Feed Forward Neural Networks (FFNN).
    - Embeddings.
    """
    def __init__(
        self,
        output_size,
        num_input_features,
        hidden_dim,
        n_layers,
        ffnn_layers,
        drop_prob,
        static_dim,
        list_unic_cat,
        embedding_dims,
        embeddings_dropout,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
    ):
        """
        Initializes the Hybrid Model.

        Parameters
        ----------
        output_size : int
            The size of the output vector.
        num_input_features : int
            The number of variables in the time series.
        hidden_dim : int
            The size of the LSTM hidden state.
        n_layers : int
            The number of layers in the LSTM.
        ffnn_layers : int
            The number of layers in the FFNN processing the combined data.
        drop_prob : float
            The dropout probability for the LSTM.
        static_dim : int
            The size of the static data.
        list_unic_cat : list
            The list of unique categories for each categorical variable.
        embedding_dims : list
            The size of the embeddings for each categorical variable.
        embeddings_dropout : float
            The dropout probability for the embeddings.
        ablation_TS : bool
            Ablation for the time series data.
        ablation_tabular : bool
            Ablation for the tabular data.
        ablation_attention : bool
            Ablation for the attention mechanism.
        
        Returns
        -------
        out : torch.Tensor
            The output of the model.
        hidden : tuple
            The hidden state and cell state of the LSTM.
        """
        super(HybridModel, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.ablation_TS = ablation_TS
        self.ablation_tabular = ablation_tabular
        self.ablation_attention = ablation_attention

        if not self.ablation_tabular:
            self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_embeddings=i, embedding_dim=dimension)
                        for i, dimension in zip(list_unic_cat, embedding_dims)
                    ]
                )
            self.embeddings_dropout = nn.Dropout(embeddings_dropout)
            self.after_embeddings = nn.Sequential(nn.Linear(sum(embedding_dims), 7), nn.ReLU())

        if not self.ablation_TS:
            self.lstm = nn.LSTM(
                num_input_features,
                hidden_dim,
                n_layers,
                dropout=drop_prob,
                batch_first=True,
            )
            self.attention = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(drop_prob)

        input_size = hidden_dim * 2 + static_dim + 7
        intermediate_size = hidden_dim
        exit_size = hidden_dim

        if ((self.ablation_TS and self.ablation_tabular)):
            raise ValueError("You cannot use this combination of ablations. It would break the model architecture.")

        if self.ablation_TS:
            input_size = static_dim + 7
            intermediate_size = static_dim + 7
            exit_size = static_dim + 7

        elif self.ablation_tabular:
            input_size = hidden_dim * 2
            intermediate_size = hidden_dim
            exit_size = hidden_dim

        if self.ablation_attention and self.ablation_tabular:
            input_size = hidden_dim
            intermediate_size = hidden_dim
            exit_size = hidden_dim

        elif self.ablation_attention and not self.ablation_tabular:
            input_size = hidden_dim + static_dim + 7
            intermediate_size = hidden_dim + static_dim + 7
            exit_size = hidden_dim + static_dim + 7

        self.fflayers = []
        for i in range(ffnn_layers - 1):
            if i == 0:
                self.fflayers.append(nn.Linear(input_size, intermediate_size))
            else:
                self.fflayers.append(nn.Linear(intermediate_size, exit_size))
        self.fflayers = nn.ModuleList(self.fflayers)
        self.final = nn.Linear(exit_size, output_size)

    def forward(self, x, hidden, static, cat):
        batch_size = x.size(0)
        x = x.to(dtype=torch.float32)
        static = static.to(dtype=torch.float32)

        if not self.ablation_TS:
            if self.ablation_attention:
                lstm_out, hidden = self.lstm(x, hidden)
                last_hidden = lstm_out[:, -1, :]
                out = self.dropout(last_hidden)
            else:
                lstm_out, hidden = self.lstm(x, hidden)
                last_hidden = lstm_out[:, -1, :]
                attn_weights = F.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attn_weights * lstm_out, dim=1)
                att_out = torch.cat((context_vector, last_hidden), 1)
                out = self.dropout(att_out)

        if not self.ablation_tabular:
            embeddings = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
            cat = torch.cat(embeddings, dim=1)
            cat = self.embeddings_dropout(cat)
            cat = self.after_embeddings(cat)
        
        if self.ablation_TS:
            for i in range(len(self.fflayers)):
                if i == 0 and static is not None:
                    out = self.fflayers[i](torch.cat((static, cat), 1))
                else:
                    out = self.fflayers[i](out)
            out = self.final(out)

        elif self.ablation_tabular:
            for i in range(len(self.fflayers)):
                out = self.fflayers[i](out)
            out = self.final(out)

        else:
            for i in range(len(self.fflayers)):
                if i == 0 and static is not None:
                    out = self.fflayers[i](torch.cat((out, static, cat), 1))
                else:
                    out = self.fflayers[i](out)
            out = self.final(out)

        out = out.view(batch_size, -1)
        return out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )
        return hidden