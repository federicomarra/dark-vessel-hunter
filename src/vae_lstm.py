import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, num_layers: int=1, z_dim: int=32, dropout: float=0.1):
        """
        Initializes the Encoder part of the VAE.
        
        Args:
            in_dim (int): Input feature dimension (e.g., 5 for lat, lon, sog, cog, etc.).
            hidden_size (int): Number of units in the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            z_dim (int): Dimension of the latent vector 'z'.
            dropout (float): Dropout probability between LSTM layers (if num_layers > 1).
        """
        super().__init__()
        
        # LSTM layer:
        # - batch_first=True means input tensors have shape (Batch, Sequence, Features)
        # - dropout is only applied if num_layers > 1
        self.lstm = nn.LSTM(
            in_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer to map the final hidden state to the mean (mu) of the latent distribution
        self.hidden_to_mu = nn.Linear(hidden_size, z_dim)
        
        # Linear layer to map the final hidden state to the log-variance (logvar) of the latent distribution
        self.hidden_to_logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x shape: (B, T, in_dim), where B=batch_size, T=seq_len
        # mask shape: (B, T)
        
        # Pass the input sequence through the LSTM
        # 'out' contains the hidden state for *every* time step
        # out shape: (B, T, H), where H=hidden_size
        out, _ = self.lstm(x)  # We ignore the final hidden/cell state '_'
        
        # --- Masked Average Pooling ---
        # We need to get a single vector summary (pooled) for each sequence in the batch.
        # We can't just take the last time step, because of padding.
        # Instead, we do a *weighted average* of all "real" time steps, using the mask.
        
        # 1. Get the denominator: the actual length of each sequence
        # .sum(dim=1) counts the number of 1s in each mask -> (B,)
        # .keepdim=True makes it (B, 1) for broadcasting
        # .clamp(min=1.0) prevents division by zero for empty sequences
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # Shape: (B, 1)

        # 2. Calculate the pooled vector
        # mask.unsqueeze(-1) changes mask from (B, T) to (B, T, 1)
        # (out * ...) zeros out the LSTM outputs for padded time steps
        # .sum(dim=1) sums all real outputs along the time dimension -> (B, H)
        # ... / denom performs the average
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom  # Shape: (B, H)
        
        # 3. Project the pooled summary vector to mu and logvar
        mu = self.hidden_to_mu(pooled)        # Shape: (B, z_dim)
        logvar = self.hidden_to_logvar(pooled)  # Shape: (B, z_dim)
        
        return mu, logvar




class DecoderLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, num_layers: int=1, z_dim: int=32, dropout: float=0.1):
        """
        Initializes the Decoder part of the VAE.
        
        Args:
            in_dim (int): Output feature dimension (must match Encoder's in_dim).
            hidden_size (int): Number of units in the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            z_dim (int): Dimension of the latent vector 'z' (must match Encoder's z_dim).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim

        # Decoder LSTM:
        # The input at each time step will be:
        # 1. The "ground truth" data from the *previous* time step (Teacher Forcing)
        # 2. The *entire* latent vector z
        # So, input_size = in_dim + z_dim
        self.lstm = nn.LSTM(
            input_size=in_dim + z_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer to transform the latent 'z' vector into a suitable
        # *initial hidden state* for the decoder LSTM
        self.z_to_hidden = nn.Linear(z_dim, hidden_size)
        
        # Linear layer to map the LSTM's output at each time step
        # from hidden_size back to the original feature dimension (in_dim)
        self.hidden_to_out = nn.Linear(hidden_size, in_dim)

    def forward(self, x:torch.Tensor, z:torch.Tensor):
        # x shape: (B, T, F), where F=in_dim
        # z shape: (B, Z), where Z=z_dim
        B, T, F = x.shape

        # --- Prepare Decoder Input (Teacher Forcing) ---
        # We need to create the "shifted" input sequence for teacher forcing.
        # The input at time 't' should be the ground truth from time 't-1'.
        
        # 1. Create a "start token" (a tensor of zeros)
        start = torch.zeros((B, 1, F), device=x.device, dtype=x.dtype)  # Shape: (B, 1, F)
        
        # 2. Get all but the last time step from the original data
        # x[:, :-1, :] -> (B, T-1, F)
        
        # 3. Concatenate the start token with the shifted sequence
        # This gives a sequence of length T, perfectly aligned.
        x_shifted = torch.cat([start, x[:, :-1, :]], dim=1)  # Shape: (B, T, F)

        # --- Prepare Latent Vector ---
        # We need to feed 'z' to the LSTM at *every* time step.
        # z.unsqueeze(1) -> (B, 1, Z)
        # .expand(B, T, Z) repeats z 'T' times without using extra memory
        z_rep = z.unsqueeze(1).expand(B, T, self.z_dim)  # Shape: (B, T, Z)

        # --- Concatenate Inputs ---
        # Combine the shifted data and the repeated 'z' vector
        dec_in = torch.cat([x_shifted, z_rep], dim=-1)  # Shape: (B, T, F + Z)

        # --- Initialize LSTM State ---
        # We initialize the LSTM's hidden/cell state using 'z'
        # This "conditions" the decoder on the latent vector.
        # .unsqueeze(0) adds the 'num_layers' dimension
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0) # Shape: (1, B, H)
        c0 = torch.zeros_like(h0)                         # Shape: (1, B, H)

        # --- Run the Decoder LSTM ---
        # We provide the combined input AND the custom initial state
        out, _ = self.lstm(dec_in, (h0, c0))  # out shape: (B, T, H)
        
        # Map the LSTM's output sequence back to the data dimension
        x_hat = self.hidden_to_out(out)       # x_hat shape: (B, T, F)
        
        return x_hat
    

class VAE_LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, z_dim: int=32, num_layers: int=1, dropout: float=0.1):
        """
        Initializes the full VAE-LSTM model.
        This class just acts as a wrapper to connect the Encoder and Decoder.
        """
        super().__init__()
        self.encoder = EncoderLSTM(in_dim, hidden_size, num_layers, z_dim, dropout)
        self.decoder = DecoderLSTM(in_dim, hidden_size, num_layers, z_dim, dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x shape: (B, T, in_dim)
        # mask shape: (B, T)
        
        # 1. Encode the input sequence to get latent distribution parameters
        mu, logvar = self.encoder(x, mask)  # Shapes: (B, z_dim), (B, z_dim)
        
        # 2. Apply the Reparameterization Trick
        std = torch.exp(0.5 * logvar)       # Calculate standard deviation
        eps = torch.randn_like(std)         # Sample from N(0, 1)
        z = mu + eps * std                  # Sample 'z' from N(mu, std)
                                            # z shape: (B, z_dim)
        
        # 3. Decode 'z' to reconstruct the original sequence
        # We pass 'x' for teacher forcing inside the decoder
        x_hat = self.decoder(x, z)          # x_hat shape: (B, T, in_Sdim)
        
        # Return the reconstruction, mu, and logvar for the loss calculation
        return x_hat, mu, logvar