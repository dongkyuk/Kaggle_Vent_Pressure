import torch
import torch.nn as nn
from model.custom_blocks import MlpEncoder


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
    def __init__(self, input_size=1, hidden_size=256, aux=True):
        super().__init__()
        self.aux = aux

        # Embeddings for categorical features
        self.r_embedding = nn.Embedding(3, hidden_size)
        self.c_embedding = nn.Embedding(3, hidden_size)
        self.u_out_embedding = nn.Embedding(2, hidden_size)

        # Individual Feature Encoders        
        self.u_in_encoder = MlpEncoder(
            input_size, hidden_size//2, hidden_size, 0, 'swish', 'ln')
        self.timestep_encoder = MlpEncoder(
            input_size, hidden_size//2, hidden_size, 0, 'swish', 'ln')

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=6)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, hidden_size))
        self.merge_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=6)

        # LSTM
        self.lstm_head = LstmHead(hidden_size)

        # Auxilary Classifiers
        if self.aux:
            self.mlp_aux1 = MlpEncoder(hidden_size*4, hidden_size*2, hidden_size, 4, 'swish', 'ln')
            self.aux1 = LstmHead(hidden_size)
            self.aux2 = LstmHead(hidden_size)

    def forward(self, x):
        # Encode individual features
        r_feat = self.r_embedding(x['rs'].int())
        c_feat = self.c_embedding(x['cs'].int())
        u_out_feat = self.u_out_embedding(x['u_outs'].int())
        u_in_feat = self.u_in_encoder(x['u_ins'].unsqueeze(-1))

        # Fuse the features with transformer encoder
        merge_tokens = self.merge_token.repeat(
            x['rs'].size(0), x['rs'].size(1), 1)
        feat = torch.stack([merge_tokens, r_feat, c_feat,
                           u_in_feat, u_out_feat], dim=2)
        if self.aux:
            aux_feat = torch.cat([merge_tokens, r_feat, c_feat,
                                u_in_feat, u_out_feat], dim=2)
            pred_aux_1 = self.aux1(self.mlp_aux1(aux_feat))

        feat += self.pos_embedding  # [batch_size, seq_len, 6, output_dim]
        feat = feat.view(-1, feat.shape[2], feat.shape[3])
        feat = self.transformer_encoder(feat)[:, 0, :]
        feat = feat.view(-1, r_feat.shape[1], feat.shape[1])

        if self.aux:
            pred_aux_2 = self.aux2(feat)

        # Decode with transformer decoder
        timestep_feat = self.timestep_encoder(x['time_steps'].unsqueeze(-1))
        feat += timestep_feat  # [batch_size, seq_len, output_dim]
        feat = self.transformer_decoder(feat)

        # Pass sequence of fused features to LSTM
        pred = self.lstm_head(feat)

        if self.aux:
            return pred, pred_aux_1, pred_aux_2

        return pred

# class TransformerRnn(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Mlp Encoders
#         self.mlp_encoder = MlpEncoder(4, 128, 256, 2)

#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         # self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6)
#         # self.pos_embedding = nn.Parameter(torch.randn(1, 80, 256))

#         # LSTM
#         # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

#         # Regression Head
#         self.regression_head = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )


#     def forward(self, x):
#         # Encode individual features
#         feat = torch.stack([x['rs'], x['cs'], x['u_ins'], x['u_outs']], dim=2) # [batch_size, seq_len, 5]
#         feat = self.mlp_encoder(feat) # [batch_size, seq_len, output_dim]

#         # Fuse the features with transformer encoder
#         feat += x['time_steps'].unsqueeze(-1) # [batch_size, seq_len, output_dim]
#         feat = self.transformer_encoder(feat) # [batch_size, seq_len, output_dim]
#         feat += x['time_steps'].unsqueeze(-1) # [batch_size, seq_len, output_dim]
#         feat = self.transformer_decoder(feat) # [batch_size, seq_len, output_dim]

#         # Pass sequence of fused features to LSTM
#         # feat = self.lstm(feat)[0] # [batch_size, seq_len, output_dim*2]

#         # Regression Head
#         pred = self.regression_head(feat).squeeze() # [batch_size, seq_len]

#         return pred


class TransformerLstm(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, aux=True):
        super().__init__()
        self.aux = aux
        self.u_out_embedding = nn.Embedding(2, 1)

        # Individual Feature Encoders
        indvidual_feat_encoder_args = dict(
            input_size=input_size,
            hidden_size=int(hidden_size/2),
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.r_encoder = nn.LSTM(**indvidual_feat_encoder_args)
        self.c_encoder = nn.LSTM(**indvidual_feat_encoder_args)
        self.u_in_encoder = nn.LSTM(**indvidual_feat_encoder_args)
        self.u_out_encoder = nn.LSTM(**indvidual_feat_encoder_args)
        self.timestep_encoder = nn.LSTM(**indvidual_feat_encoder_args)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=6)
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, hidden_size))
        self.merge_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # LSTM
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size*2),
            nn.Linear(hidden_size*2, 1),
        )

        # Auxiliary layers
        self.fuse_feat_encoder = MlpEncoder(
            hidden_size*5, hidden_size*2, hidden_size, 5)
        self.lstm_aux = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                num_layers=2, batch_first=True, bidirectional=True)
        self.regression_head_aux = nn.Sequential(
            nn.LayerNorm(hidden_size*2),
            nn.Linear(hidden_size*2, 1),
        )

    def forward(self, x):
        # Encode individual features
        r_feat = self.r_encoder(x['rs'].unsqueeze(-1))[0]
        c_feat = self.c_encoder(x['cs'].unsqueeze(-1))[0]
        u_in_feat = self.u_in_encoder(x['u_ins'].unsqueeze(-1))[0]
        u_out_feat = self.u_out_encoder(
            self.u_out_embedding(x['u_outs'].unsqueeze(-1)))[0]
        timestep_feat = self.timestep_encoder(x['time_steps'].unsqueeze(-1))[0]

        # Fuse the features with transformer encoder
        merge_tokens = self.merge_token.repeat(
            x['rs'].size(0), x['rs'].size(1), 1)
        feat = torch.stack([merge_tokens, r_feat, c_feat,
                           u_in_feat, u_out_feat, timestep_feat], dim=2)
        feat += self.pos_embedding  # [batch_size, seq_len, 6, output_dim]
        feat = feat.view(-1, feat.shape[2], feat.shape[3])
        feat = self.transformer_encoder(feat)[:, 0, :]
        feat = feat.view(-1, r_feat.shape[1], feat.shape[1])

        # Pass sequence of fused features to LSTM
        feat = self.lstm(feat)[0]  # [batch_size, seq_len, output_dim*2]

        # Regression Head
        pred = self.regression_head(feat).squeeze()  # [batch_size, seq_len]

        if self.aux:
            feat = torch.cat([r_feat, c_feat, u_in_feat,
                             u_out_feat, timestep_feat], dim=2)
            feat = self.fuse_feat_encoder(feat)
            feat = self.lstm_aux(feat)[0]
            pred_aux = self.regression_head_aux(feat).squeeze()
            return pred, pred_aux

        return pred


class SimpleLstm(nn.Module):
    def __init__(self):
        super().__init__()
        # Mlp Encoders
        self.r_encoder = MlpEncoder(1, 128, 256, 5)
        self.c_encoder = MlpEncoder(1, 128, 256, 5)
        self.u_in_encoder = MlpEncoder(1, 128, 256, 5)
        self.u_out_encoder = MlpEncoder(1, 128, 256, 5)
        self.timestep_encoder = MlpEncoder(1, 128, 256, 5)
        # self.r_encoder = nn.LSTM(input_size=1, hidden_size=4, num_layers=2, batch_first=True, bidirectional=True)
        # self.c_encoder = nn.LSTM(input_size=1, hidden_size=4, num_layers=2, batch_first=True, bidirectional=True)
        # self.u_in_encoder = nn.LSTM(input_size=1, hidden_size=4, num_layers=2, batch_first=True, bidirectional=True)
        # self.u_out_encoder = nn.LSTM(input_size=1, hidden_size=4, num_layers=2, batch_first=True, bidirectional=True)
        # self.timestep_encoder = nn.LSTM(input_size=1, hidden_size=4, num_layers=2, batch_first=True, bidirectional=True)

        self.fuse_feat_encoder = MlpEncoder(1280, 512, 256, 2)

        # LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=2, batch_first=True, bidirectional=True)

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Encode individual features
        # feat = torch.stack([x['rs'], x['cs'], x['u_ins'], x['u_outs'], x['time_steps']], dim=2) # [batch_size, seq_len, 5]
        # [batch_size, seq_len, output_dim]
        r_feat = self.r_encoder(x['rs'].unsqueeze(-1))
        c_feat = self.c_encoder(x['cs'].unsqueeze(-1))
        u_in_feat = self.u_in_encoder(x['u_ins'].unsqueeze(-1))
        u_out_feat = self.u_out_encoder(x['u_outs'].unsqueeze(-1))
        timestep_feat = self.timestep_encoder(x['time_steps'].unsqueeze(-1))

        # Fuse the features with mlp encoder
        # [batch_size, seq_len, 5*output_dim]
        feat = torch.cat([r_feat, c_feat, u_in_feat,
                         u_out_feat, timestep_feat], dim=-1)
        # [batch_size, seq_len, output_dim]
        feat = self.fuse_feat_encoder(feat)

        # Pass sequence of fused features to LSTM
        feat = self.lstm(feat)[0]  # [batch_size, seq_len, output_dim*2]

        # Regression Head
        pred = self.regression_head(feat).squeeze()  # [batch_size, seq_len]

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
