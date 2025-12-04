import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x, lengths):
        # Pack sequence to ignore padding computation
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Encoder outputs: output, (hidden, cell)
        # hidden shape: (num_layers, batch, hidden_dim)
        _, (hidden, cell) = self.lstm(packed_x)
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, num_layers, num_ship_types, shiptype_emb_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Ship Type Injection
        self.shiptype_embedding = nn.Embedding(num_ship_types, shiptype_emb_dim)
        
        # --- THE BRIDGE ---
        # We take (Latent_Dim + Ship_Emb_Dim) and project it back to (Hidden_Dim)
        # This initializes the Decoder's memory with the compressed track info + ship type
        self.bridge_hidden = nn.Linear(latent_dim + shiptype_emb_dim, hidden_dim)
        self.bridge_cell = nn.Linear(latent_dim + shiptype_emb_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, z, shiptypes):
        # x: (Batch, Seq_Len, Features)
        # z: (Batch, Latent_Dim) -> The compressed track representation
        # shiptypes: (Batch)
        
        # 1. Get Ship Embeddings
        ship_emb = self.shiptype_embedding(shiptypes) # (Batch, Emb_Dim)
        
        # 2. Concatenate Latent Code + Ship Info
        combined_features = torch.cat((z, ship_emb), dim=1) # (Batch, Latent + Emb)
        
        # 3. Project to Initialize Decoder State
        # We create the initial hidden/cell state for *all* layers of the decoder
        # (Batch, Hidden) -> (1, Batch, Hidden) -> Repeat for num_layers
        init_hidden = torch.tanh(self.bridge_hidden(combined_features))
        init_cell = torch.tanh(self.bridge_cell(combined_features))
        
        # Shape: (Num_Layers, Batch, Hidden)
        init_hidden = init_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        init_cell = init_cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 4. LSTM Forward
        # We feed x (the sequence) into the decoder (Teacher Forcing)
        output, _ = self.lstm(x, (init_hidden, init_cell))
        
        # 5. Prediction
        prediction = self.fc_out(output)
        
        return prediction

class AIS_LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, num_ship_types, shiptype_emb_dim, dropout=0.0):
        super(AIS_LSTM_Autoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        
        # --- BOTTLENECK LAYER ---
        # Compresses the Encoder's final hidden state down to 'latent_dim'
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, num_layers, num_ship_types, shiptype_emb_dim, dropout)
        
    def forward(self, x, lengths, shiptypes):
        # 1. Encode
        # hidden shape: (Num_Layers, Batch, Hidden_Dim)
        hidden, _ = self.encoder(x, lengths)
        
        # 2. Bottleneck
        # We take the *last layer* of the encoder's hidden state as the representation
        last_layer_hidden = hidden[-1] # (Batch, Hidden_Dim)
        
        # Project to Latent Space
        z = self.hidden_to_latent(last_layer_hidden) # (Batch, Latent_Dim)
        
        # 3. Decode
        # We pass the Latent Vector 'z' and ship info to the decoder
        reconstructed = self.decoder(x, z, shiptypes)
        
        return reconstructed