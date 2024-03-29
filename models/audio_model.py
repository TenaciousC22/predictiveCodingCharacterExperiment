# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
##############################################################################
# Minimal code to load a CPC checkpoint
##############################################################################
from custom_layers import EqualizedConv1d
from util.seq_alignment import collapseLabelChain


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x

class ShiftedConv(nn.Module):
    def __init__(self, dimOutputAR, dimOutputEncoder, kernelSize):
        super(ShiftedConv, self).__init__()
        self.module = EqualizedConv1d(dimOutputAR, dimOutputEncoder,
                                      kernelSize, equalized=True,
                                      padding=0)
        self.kernelSize = kernelSize

    def forward(self, x):

        # Input format: N, S, C -> need to move to N, C, S
        N, S, C = x.size()
        x = x.permute(0, 2, 1)

        padding = torch.zeros(N, C, self.kernelSize - 1, device=x.device)
        x = torch.cat([padding, x], dim=2)
        x = self.module(x)
        x = x.permute(0, 2, 1)
        return x

class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512):

        super(CPCEncoder, self).__init__()
        normLayer = ChannelNorm

        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x

class CPCAudioVisualEncoder(nn.Module):
    def __init__(self,
                 sizeHidden=512, sizeVisualFeature=512):

        super(CPCAudioVisualEncoder, self).__init__()
        normLayer = ChannelNorm

        self.audio_conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.audio_batchNorm0 = normLayer(sizeHidden)
        self.audio_conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.audio_batchNorm1 = normLayer(sizeHidden)
        self.audio_conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.audio_batchNorm2 = normLayer(sizeHidden)
        self.audio_conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.audio_batchNorm3 = normLayer(sizeHidden)
        self.audio_conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.audio_batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

        self.visual_conv0 = nn.Conv1d(sizeVisualFeature, sizeHidden, 3)
        self.visual_batchNorm0 = normLayer(sizeHidden)
        self.visual_conv1 = nn.Conv1d(sizeHidden, sizeHidden, 3)
        self.visual_batchNorm1 = normLayer(sizeHidden)
        self.visual_conv2 = nn.Conv1d(sizeHidden, sizeHidden, 3)
        self.visual_batchNorm2 = normLayer(sizeHidden)
        self.visual_conv3 = nn.Conv1d(sizeHidden, sizeHidden, 3)
        self.visual_batchNorm3 = normLayer(sizeHidden)
        self.visual_conv4 = nn.Conv1d(sizeHidden, sizeHidden, 3)
        self.visual_batchNorm4 = normLayer(sizeHidden)

        self.audio_visual_conv0 = nn.Conv1d(sizeHidden*2, sizeHidden, 1)
        self.audio_visual_batchNorm0 = normLayer(sizeHidden)

class CPCAudioVisualConditioningEncoder(nn.Module):
    def __init__(self,
                 sizeHidden=512, sizeVisualFeature=512):
        super(CPCAudioVisualConditioningEncoder, self).__init__()
        normLayer = ChannelNorm

        self.audio_conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.audio_batchNorm0 = normLayer(sizeHidden)
        self.audio_conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.audio_batchNorm1 = normLayer(sizeHidden)
        self.audio_conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                                     stride=2, padding=1)
        self.audio_batchNorm2 = normLayer(sizeHidden)
        self.audio_conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.audio_batchNorm3 = normLayer(sizeHidden)
        self.audio_conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.audio_batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

        self.visual_conv0 = nn.ConvTranspose1d(sizeVisualFeature, sizeHidden, 1)
        self.visual_conv1 = nn.ConvTranspose1d(sizeVisualFeature, sizeHidden, 1)
        self.visual_conv2 = nn.ConvTranspose1d(sizeVisualFeature, sizeHidden, 1)
        self.visual_conv3 = nn.ConvTranspose1d(sizeVisualFeature, sizeHidden, 1)
        self.visual_conv4 = nn.ConvTranspose1d(sizeVisualFeature, sizeHidden, 1)


    def getDimOutput(self):
        return self.audio_conv4.out_channels

    def forward(self, x_inputs):
        audio_x, visual_x = x_inputs
        x = audio_x

        x = self.audio_batchNorm0(self.audio_conv0(x))
        x += self.visual_conv0(F.interpolate(visual_x, size=x.shape[2]))
        x = F.relu(x)

        x = self.audio_batchNorm1(self.audio_conv1(x))
        x += self.visual_conv1(F.interpolate(visual_x, size=x.shape[2]))
        x = F.relu(x)

        x = self.audio_batchNorm2(self.audio_conv2(x))
        x += self.visual_conv2(F.interpolate(visual_x, size=x.shape[2]))
        x = F.relu(x)

        x = self.audio_batchNorm3(self.audio_conv3(x))
        x += self.visual_conv3(F.interpolate(visual_x, size=x.shape[2]))
        x = F.relu(x)

        x = self.audio_batchNorm4(self.audio_conv4(x))
        x += self.visual_conv4(F.interpolate(visual_x, size=x.shape[2]))
        x = F.relu(x)

        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU):

        super(CPCAR, self).__init__()
        self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                               num_layers=nLevelsGRU, batch_first=True)
        self.hidden = None
        self.keepHidden = keepHidden

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()
        return x


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR

        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            if rnnMode == 'RNN':
                self.predictors.append(
                    nn.RNN(dimOutputAR, dimOutputEncoder))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'LSTM':
                self.predictors.append(
                    nn.LSTM(dimOutputAR, dimOutputEncoder, batch_first=True))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'conv4':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 4))
            elif rnnMode == 'conv8':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 8))
            elif rnnMode == 'conv12':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 12))
            elif rnnMode == 'transformer':
                from transformers import buildTransformerAR
                self.predictors.append(
                    buildTransformerAR(dimOutputEncoder,
                                       1,
                                       sizeInputSeq,
                                       False))
            else:
                self.predictors.append(
                    nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
                if dimOutputEncoder > dimOutputAR:
                    residual = dimOutputEncoder - dimOutputAR
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                        dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c, candidates):

        assert(len(candidates) == len(self.predictors))
        out = []

        # UGLY
        if isinstance(self.predictors[0], EqualizedConv1d):
            c = c.permute(0, 2, 1)

        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            if isinstance(self.predictors[k], EqualizedConv1d):
                locC = locC.permute(0, 2, 1)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC*candidates[k]).mean(dim=3)
            out.append(outK)
        return out


class BaseCriterion(nn.Module):

    def warmUp(self):
        return False

    def update(self):
        return

class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,             # Number of steps
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 mode=None,
                 rnnMode=False,
                 dropout=False,
                 speakerEmbedding=0,
                 nSpeakers=0,
                 sizeInputSeq=128):

        super(CPCUnsupersivedCriterion, self).__init__()
        if speakerEmbedding > 0:
            print(
                f"Using {speakerEmbedding} speaker embeddings for {nSpeakers} speakers")
            self.speakerEmb = torch.nn.Embedding(nSpeakers, speakerEmbedding)
            dimOutputAR += speakerEmbedding
        else:
            self.speakerEmb = None

        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder, rnnMode=rnnMode,
            dropout=dropout, sizeInputSeq=sizeInputSeq - nPredicts)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode

    def sampleClean(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        batchIdx = torch.randint(low=0, high=batchSize,
                                 size=(self.negativeSamplingExt
                                       * windowSize * batchSize, ),
                                 device=encodedData.device)

        seqIdx = torch.randint(low=1, high=nNegativeExt,
                               size=(self.negativeSamplingExt
                                     * windowSize * batchSize, ),
                               device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
        seqIdx += baseIdx.contiguous().view(-1)
        seqIdx = torch.remainder(seqIdx, nNegativeExt)

        extIdx = seqIdx + batchIdx * nNegativeExt
        negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                     windowSize, dimEncoded)

        labelLoss = torch.zeros((batchSize * windowSize),
                                dtype=torch.long,
                                device=encodedData.device)

        for k in range(1, self.nPredicts + 1):

            # Positive samples
            if k < self.nPredicts:
                posSeq = encodedData[:, k:-(self.nPredicts-k)]
            else:
                posSeq = encodedData[:, k:]

            posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
            fullSeq = torch.cat((posSeq, negExt), dim=1)
            outputs.append(fullSeq)

        return outputs, labelLoss

    def getInnerLoss(self):

        return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

    def forward(self, cFeature, encodedData, label):

        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nPredicts

        cFeature = cFeature[:, :windowSize]

        sampledData, labelLoss = self.sampleClean(encodedData, windowSize)

        if self.speakerEmb is not None:
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)
            embeddedSpeaker = self.speakerEmb(l_)
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)

        predictions = self.wPrediction(cFeature, sampledData)

        outLosses = [0 for x in range(self.nPredicts)]
        outAcc = [0 for x in range(self.nPredicts)]

        for k, locPreds in enumerate(predictions[:self.nPredicts]):
            locPreds = locPreds.permute(0, 2, 1)
            locPreds = locPreds.contiguous().view(-1, locPreds.size(2))
            lossK = self.lossCriterion(locPreds, labelLoss)
            outLosses[k] += lossK.view(1, -1)
            _, predsIndex = locPreds.max(1)
            outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

        return torch.cat(outLosses, dim=1), torch.cat(outAcc, dim=1) / (windowSize * batchSize)

class CTCOldPhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder):

        super(CTCOldPhoneCriterion, self).__init__()
        self.PhoneCriterionClassifier = nn.Linear(dimEncoder, nPhones + 1)
        self.lossCriterion = nn.CTCLoss(blank=nPhones, zero_infinity=True)
        self.onEncoder = onEncoder
        if onEncoder:
            raise ValueError("On encoder version not implemented yet")
        self.BLANK_LABEL = nPhones

    def getPrediction(self, cFeature):
        B, S, H = cFeature.size()
        cFeature = cFeature.contiguous().view(B*S, H)
        return self.PhoneCriterionClassifier(cFeature).view(B, S, -1)

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        B, S, H = cFeature.size()
        predictions = self.getPrediction(cFeature)

        label = label.to(predictions.device)
        label,  sizeLabels = collapseLabelChain(label)

        avgPER = 0.
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        targetSizePred = torch.ones(B, dtype=torch.int64,
                                    device=predictions.device) * S
        loss = self.lossCriterion(predictions, label,
                                  targetSizePred, sizeLabels).view(1, -1)

        return loss, avgPER * torch.ones(1, 1, device=loss.device)

def cutData(seq, sizeSeq):
    maxSeq = sizeSeq.max()
    seq = seq[:, :maxSeq]
    return seq

class CTCPhoneCriterion(torch.nn.Module):

    def __init__(self, dimEncoder, nPhones, LSTM=False, sizeKernel=8,
                 seqNorm=False, dropout=False, reduction='mean'):

        super(CTCPhoneCriterion, self).__init__()
        self.seqNorm = seqNorm
        self.epsilon = 1e-8
        self.dropout = torch.nn.Dropout2d(
            p=0.5, inplace=False) if dropout else None
        self.conv1 = torch.nn.LSTM(dimEncoder, dimEncoder,
                                   num_layers=1, batch_first=True)
        self.PhoneCriterionClassifier = torch.nn.Conv1d(
            dimEncoder, nPhones + 1, sizeKernel, stride=sizeKernel // 2)
        self.lossCriterion = torch.nn.CTCLoss(blank=nPhones,
                                              reduction=reduction,
                                              zero_infinity=True)
        self.relu = torch.nn.ReLU()
        self.BLANK_LABEL = nPhones
        self.useLSTM = LSTM

    def getPrediction(self, cFeature):
        B, S, H = cFeature.size()
        if self.seqNorm:
            m = cFeature.mean(dim=1, keepdim=True)
            v = cFeature.var(dim=1, keepdim=True)
            cFeature = (cFeature - m) / torch.sqrt(v + self.epsilon)
        if self.useLSTM:
            cFeature = self.conv1(cFeature)[0]

        cFeature = cFeature.permute(0, 2, 1)

        if self.dropout is not None:
            cFeature = self.dropout(cFeature)

        return self.PhoneCriterionClassifier(cFeature).permute(0, 2, 1)

    def forward(self, cFeature, featureSize, label, labelSize):

        # cFeature.size() : batchSize x seq Size x hidden size
        B, S, H = cFeature.size()
        predictions = self.getPrediction(cFeature)
        featureSize //= 4
        predictions = cutData(predictions, featureSize)
        featureSize = torch.clamp(featureSize, max=predictions.size(1))
        label = cutData(label, labelSize)
        if labelSize.min() <= 0:
            print(label, labelSize)
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        loss = self.lossCriterion(predictions, label,
                                  featureSize, labelSize).view(1, -1)

        if torch.isinf(loss).sum() > 0 or torch.isnan(loss).sum() > 0:
            loss = 0

        return loss

class FBAudioCPCLightning(pl.LightningModule):
    def __init__(self, dim_size=256, pred_steps=12, negative_samples=128, batch_size=8, pred_rnn_mode="transformer", seq_len=20480, encoder="audio"):
        super().__init__()
        self.dim_size = dim_size
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.pred_rnn_mode = pred_rnn_mode

        if encoder == 'audio':
            self.encoder = CPCEncoder(sizeHidden=dim_size)

        #outdated
        elif encoder == 'audiovisual':
            self.encoder = CPCAudioVisualEncoder(sizeHidden=dim_size)

        #outdated
        elif encoder == 'audiovisual_conditioning':
            self.encoder = CPCAudioVisualConditioningEncoder(sizeHidden=dim_size)

        self.ar = CPCAR(dim_size, dim_size, False, 1)
        self.cpc_model = CPCModel(self.encoder, self.ar)
        self.cpc_criterion = CPCUnsupersivedCriterion(pred_steps, dim_size, dim_size, self.negative_samples, rnnMode=self.pred_rnn_mode)



    def training_step(self, x, batch_idx):
        nce_loss, nce_acc = self.shared_step(x, batch_idx)
        self.log("train_loss", nce_loss)
        return nce_loss

    def validation_step(self, x, batch_idx):
        nce_loss, nce_acc  = self.shared_step(x, batch_idx)
        self.log("val_loss", nce_loss)
        return nce_loss

    def test_step(self, x, batch_idx):
        nce_loss, nce_acc  = self.shared_step(x, batch_idx)
        self.log("test_loss", nce_loss)
        return nce_loss

    def shared_step(self, x, batch_idx):
        cFeature, encodedData, label = self.cpc_model(x, None)

        allLosses, allAcc = self.cpc_criterion(cFeature, encodedData, label)

        loss = allLosses.sum()
        acc = allAcc.mean()

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
        return optimizer

    def embedding(self, x, context=False, norm=True):
        cFeature, encodedData, label = self.cpc_model(x, None)

        if context:
            embedding = cFeature
        else:
            embedding = encodedData

        if norm:
            mean = embedding.mean(dim=1, keepdim=True)
            var = embedding.var(dim=1, keepdim=True)
            embedding = (embedding - mean) / torch.sqrt(var + 1e-08)

        return embedding

class FeatureModule(torch.nn.Module):
    r"""
    A simpler interface to handle CPC models. Useful for a smooth workflow when
    working with CPC trained features.
    """

    def __init__(self, featureMaker, get_encoded,
                 seq_norm=True):
        super(FeatureModule, self).__init__()
        self.get_encoded = get_encoded
        self.model = featureMaker
        self.seq_norm = seq_norm
        self.config = None

    def forward(self, batch_data):
        # Input Size : BatchSize x 1 x SeqSize
        # Feature size: BatchSize x SeqSize x ChannelSize
        # if self.is_cuda:
        batch_data = batch_data.cuda()
        cFeature, encoded, _ = self.model(batch_data, None)
        if self.get_encoded:
            cFeature = encoded
        if self.seq_norm:
            mean = cFeature.mean(dim=1, keepdim=True)
            var = cFeature.var(dim=1, keepdim=True)
            cFeature = (cFeature - mean) / torch.sqrt(var + 1e-08)
        return cFeature

    def cuda(self):
        self.is_cuda = True
        super(FeatureModule, self).cuda()

    def cpu(self):
        self.is_cuda = False
        super(FeatureModule, self).cuda()

    def get_output_dim(self):
        if self.get_encoded:
            return self.config["hiddenEncoder"]
        return self.config["hiddenGar"]

class FBAudioCPCPhonemeClassifierLightning(pl.LightningModule):
    def __init__(self, src_checkpoint_path, dim_size=256, batch_size=8, encoder="audio", cached=True, LSTM=False):
        super().__init__()
        self.dim_size = dim_size
        self.batch_size = batch_size
        self.cached = True

        if encoder == 'audio':
            self.encoder = CPCEncoder(sizeHidden=dim_size)

        #outdated
        elif encoder == 'audiovisual':
            self.encoder = CPCAudioVisualEncoder(sizeHidden=dim_size)

        #outdated
        elif encoder == 'audiovisual_conditioning':
            self.encoder = CPCAudioVisualConditioningEncoder(sizeHidden=dim_size)

        self.ar = CPCAR(dim_size, dim_size, False, 1)
        self.cpc_model = CPCModel(self.encoder, self.ar)
        self.phoneme_criterion = CTCPhoneCriterion(self.dim_size, 43, LSTM=LSTM)

        checkpoint = torch.load(src_checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'], strict=False)

        self.cpc_model.eval()

        for g in self.cpc_model.parameters():
            g.requires_grad = False


    def training_step(self, x, batch_idx):
        ctc_loss = self.shared_step(x, batch_idx)
        self.log("train_loss", ctc_loss)

        return ctc_loss

    def validation_step(self, x, batch_idx):
        ctc_loss = self.shared_step(x, batch_idx)
        self.log("val_loss", ctc_loss)

        return ctc_loss

    def test_step(self, x, batch_idx):
        ctc_loss = self.shared_step(x, batch_idx)
        self.log("test_loss", ctc_loss)

        return ctc_loss

    def shared_step(self, data, batch_idx):
        x, x_len, label, label_len = data

        if not self.cached:
            cFeature, encodedData, label = self.cpc_model(x, label)
            x_len //= 160
        else:
            cFeature = x

        # mean = cFeature.mean(dim=1, keepdim=True)
        # var = cFeature.var(dim=1, keepdim=True)
        # cFeature = (cFeature - mean) / torch.sqrt(var + 1e-08)

        allLosses = self.phoneme_criterion(cFeature, x_len, label, label_len)

        loss = allLosses.sum()

        return loss

    def get_predictions(self, x):
        cFeature, encodedData, label = self.cpc_model(x, None)
        predictions = torch.nn.functional.softmax(self.phoneme_criterion.getPrediction(cFeature), dim=2)

        return predictions

    def configure_optimizers(self):
        g_params = list(self.phoneme_criterion.parameters())
        optimizer = torch.optim.Adam(g_params, lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
        return optimizer