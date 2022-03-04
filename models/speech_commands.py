import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.audio_model import ChannelNorm

class GRUInference(nn.Module):
    def __init__(self, n_classes=30, dim=128, input_dim=256):
        super(GRUInference, self).__init__()

        self.dim = dim

        normLayer = ChannelNorm

        self.conv0 = nn.Conv1d(1, input_dim, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(input_dim)
        self.conv1 = nn.Conv1d(input_dim, input_dim, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(input_dim)
        self.conv2 = nn.Conv1d(input_dim, input_dim, 4, stride=2, padding=1)
        self.batchNorm2 = normLayer(input_dim)
        self.conv3 = nn.Conv1d(input_dim, input_dim, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(input_dim)
        self.conv4 = nn.Conv1d(input_dim, input_dim, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(input_dim)
        self.DOWNSAMPLING = 160

        self.gru = nn.GRU(input_size=input_dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.dim, n_classes)

    def forward(self, x):

        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))

        x, hidden = self.gru(x.permute((0, 2, 1)))
        x = self.fc(hidden)
        x = torch.squeeze(x)
        return x


class GRUClassifier(nn.Module):
    def __init__(self, n_classes=30, dim=128, input_dim=256):
        super(GRUClassifier, self).__init__()

        self.dim = dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.dim, n_classes)

    def forward(self, x):
        x, hidden = self.gru(x)
        x = self.fc(hidden)
        x = torch.squeeze(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.squeeze(output.mean(1))
        output = self.decoder(output)

        return output



class GRUMAMLClassifier(nn.Module):
    def __init__(self, n_classes=5, dim=128, input_dim=256):
        super(GRUMAMLClassifier, self).__init__()

        self.dim = dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.dim, self.dim*2)
        self.fc = nn.Linear(self.dim*2, n_classes)

    def forward(self, x):
        x, hidden = self.gru(x)
        x = x[:,-1, :].squeeze(1)
        x = self.linear(x)
        x = self.fc(x)
        x = torch.squeeze(x)

        return x

class ConvClassifier(nn.Module):
    def __init__(self, n_classes=30, dim=128, input_dim=256):
        super(ConvClassifier, self).__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(1, 64, (20, 8))
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv2d(64, 64, (10, 4))
        self.pool2 = nn.MaxPool2d((1, 1))
        self.lin = nn.Linear(368640, 32)
        self.dnn = nn.Linear(32, 128)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))  # shape: (batch, channels, i1, o1)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))  # shape: (batch, o1, i2, o2)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # shape: (batch, o3)
        x = self.lin(x)
        x = self.dnn(x)
        x = F.relu(x)
        return self.out(x)


class SpeechCommandsCPCClassificationLightning(pl.LightningModule):
    def __init__(self, n_classes=30, batch_size=8, classifier_type='gru'):
        super().__init__()
        self.batch_size = batch_size
        if classifier_type == 'gru':
            self.classifier = GRUClassifier(n_classes)
        elif classifier_type == 'conv':
            self.classifier = ConvClassifier(n_classes)
        elif classifier_type == 'transformer':
            self.classifier = TransformerClassifier(n_classes, 256, 8, 256, 2)


        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def training_step(self, data, batch_idx):
        x, y = data
        loss, y_hat = self.shared_step(x, y, batch_idx)
        self.log("train_loss", loss)
        self.train_acc(F.softmax(y_hat, dim=-1), y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, data, batch_idx):
        x, y = data
        loss, y_hat = self.shared_step(x, y, batch_idx)
        self.log("val_loss", loss)

        self.valid_acc(F.softmax(y_hat, dim=-1), y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, data, batch_idx):
        x, y = data
        loss, y_hat = self.shared_step(x, y, batch_idx)
        self.log("test_loss", loss)

        self.test_acc(F.softmax(y_hat, dim=-1), y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

        return loss

    def shared_step(self, x, y, batch_idx):
        y_hat = self.classifier(x)

        loss = self.ce_loss(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        return optimizer

    def classification(self, x):
        res = self.classifier(x)
        return F.softmax(res, dim=-1)