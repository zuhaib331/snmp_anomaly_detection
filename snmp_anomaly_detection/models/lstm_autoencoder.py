from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - lets preprocessing work without torch installed
    torch = None
    nn = None


if nn is not None:

    class LSTMAutoencoder(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            latent_size: int = 32,
            num_layers: int = 1,
        ):
            super().__init__()
            self.encoder = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                num_layers=num_layers,
            )
            self.to_latent = nn.Linear(hidden_size, latent_size)
            self.from_latent = nn.Linear(latent_size, hidden_size)
            self.decoder = nn.LSTM(
                hidden_size,
                hidden_size,
                batch_first=True,
                num_layers=num_layers,
            )
            self.output_layer = nn.Linear(hidden_size, input_size)

        def forward(self, inputs):
            _, (hidden_state, _) = self.encoder(inputs)
            latent = self.to_latent(hidden_state[-1])
            repeated = self.from_latent(latent).unsqueeze(1).repeat(1, inputs.size(1), 1)
            decoded, _ = self.decoder(repeated)
            return self.output_layer(decoded)

else:

    class LSTMAutoencoder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use LSTMAutoencoder.")
