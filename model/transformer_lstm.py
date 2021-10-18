import torch
import torch.nn as nn
from model.custom_blocks import MlpEncoder, Time2Vec


class IndiviudalFeatureEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(IndiviudalFeatureEncoder, self).__init__()
        # Embeddings for categorical features
        self.r_embedding = nn.Embedding(3, hidden_size)
        self.c_embedding = nn.Embedding(3, hidden_size)
        self.u_out_embedding = nn.Embedding(2, hidden_size)

        # Mlp Encoders
        self.u_in_encoder = MlpEncoder(
            input_size, hidden_size//2, hidden_size, 0, 'swish', 'ln')
        self.timestep_encoder = Time2Vec(input_size, hidden_size)
        # self.timestep_encoder = MlpEncoder(
        #     input_size, hidden_size//2, hidden_size, 0, 'swish', 'ln')

    def forward(self, x):
        r_feat = self.r_embedding(x['rs'].int())
        c_feat = self.c_embedding(x['cs'].int())
        u_out_feat = self.u_out_embedding(x['u_outs'].int())
        u_in_feat = self.u_in_encoder(x['u_ins'].unsqueeze(-1))
        timestep_feat = self.timestep_encoder(x['time_steps'].unsqueeze(-1))
        return r_feat, c_feat, u_out_feat, u_in_feat, timestep_feat


class LstmHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(input_size*2),
            nn.Linear(input_size*2, 1),
        )

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.regression_head(x).squeeze()
        return x


class TransformerOnly(nn.Module):
    def __init__(self, input_size=1, hidden_size=768):
        super().__init__()
        # Individual Feature Encoder
        self.indiviudal_feature_encoder = IndiviudalFeatureEncoder(
            input_size, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=12, batch_first=True, dim_feedforward=3072, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=12)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, hidden_size))
        self.merge_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=12, batch_first=True, dim_feedforward=3072, activation='gelu')
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=12)

        # LSTM
        self.lstm_head = LstmHead(hidden_size)

    def forward(self, x):
        # Encode individual features
        r_feat, c_feat, u_out_feat, u_in_feat, timestep_feat = self.indiviudal_feature_encoder(
            x)

        # Fuse the features with transformer encoder
        merge_tokens = self.merge_token.repeat(
            x['rs'].size(0), x['rs'].size(1), 1)
        feat = torch.stack([merge_tokens, r_feat, c_feat,
                           u_in_feat, u_out_feat], dim=2)
        feat += self.pos_embedding  # [batch_size, seq_len, 6, output_dim]
        feat = feat.view(-1, feat.shape[2], feat.shape[3])
        feat = self.transformer_encoder(feat)[:, 0, :]
        feat = feat.view(-1, r_feat.shape[1], feat.shape[1])

        # Decode with transformer decoder
        feat += timestep_feat  # [batch_size, seq_len, output_dim]
        feat = self.transformer_decoder(feat)

        # Pass sequence of fused features to LSTM
        pred = self.lstm_head(feat)

        return pred


class SimpleLstm(nn.Module):
    def __init__(self, input_size=1, hidden_size=768):
        super().__init__()
        # Individual Feature Encoder
        self.indiviudal_feature_encoder = IndiviudalFeatureEncoder(input_size, hidden_size)
        self.fuse_feat_encoder = MlpEncoder(int(hidden_size*5), int(hidden_size*2), hidden_size, 2)

        # LSTM
        self.lstm_head = LstmHead(hidden_size)

    def forward(self, x):
        r_feat, c_feat, u_out_feat, u_in_feat, timestep_feat = self.indiviudal_feature_encoder(x)
        feat = torch.cat([r_feat, c_feat, u_in_feat,
                         u_out_feat, timestep_feat], dim=-1)
        feat = self.fuse_feat_encoder(feat)
        pred = self.lstm_head(feat)
        return pred

# x = dict(
#     rs=torch.randn(10, 80),
#     cs=torch.randn(10, 80),
#     u_ins=torch.randn(10, 80),
#     u_outs=torch.randn(10, 80),
#     time_steps=torch.randn(10, 80),
# )

# model = TransformerRnn()
# model(x)
