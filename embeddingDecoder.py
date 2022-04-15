class embeddingDecoder(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, batch_size=8, visualFeatureDim=512, numHeads=8, numLayers=6, numLevelsGRU=1, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, LSTM=False, freeze=True):
		super(embeddingDecoder, self).__init__()
		self.dim_size = dim_size
		self.batch_size = batch_size

		self.model = decoderModel(dim_size, dim_size, keepHidden=False)

		#Applies criterion
		self.character_criterion = CTCCharacterCriterionV3(self.dim_size, 38, LSTM=LSTM)

		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

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

		x = self.model(x)
		allLosses = self.character_criterion(x, x_len, label, label_len)

		loss = allLosses.sum()
		return loss

	def get_predictions(self, x):
		cFeature = self.model(x)
		predictions = torch.nn.functional.softmax(self.character_criterion.getPrediction(cFeature), dim=2)

		return predictions

	def configure_optimizers(self):
		g_params = list(self.character_criterion.parameters())
		optimizer = torch.optim.Adam(g_params, lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
		return optimizer

class decoderModel(nn.Module):

	def __init__(self, inSize=256, dim_size=256, peMaxLen=2500, keepHidden=False, numHeads=8, numLayers=6, fcHiddenSize=2048, dropout=0.1, numClasses=39, cached=True):

		super(decoderModel, self).__init__()
		#self.baseNet = nn.LSTM(dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True)
		self.hidden = None
		self.keepHidden = keepHidden
		encoderLayer = nn.TransformerEncoderLayer(d_model=dim_size, nhead=numHeads, dim_feedforward=fcHiddenSize, dropout=dropout)

		#Declare joint decoder
		self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

		self.cached = cached

	#Feed forward function
	def forward(self, jointBatch):

		# jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
		# jointBatch = self.jointConv(jointBatch)
		jointBatch=jointBatch.transpose(1, 2)#.transpose(0, 1)

		jointBatch = self.jointDecoder(jointBatch)
		outputBatch = jointBatch.transpose(0, 1).transpose(1, 2)
		# jointBatch = self.outputConv(jointBatch)
		# jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
		# outputBatch = F.log_softmax(jointBatch, dim=2)

		return outputBatch

class CTCCharacterCriterionV3(torch.nn.Module):

	def __init__(self, dimEncoder, nPhones, LSTM=False, sizeKernel=8,
				 seqNorm=False, dropout=False, reduction='mean'):

		super(CTCCharacterCriterionV3, self).__init__()
		self.seqNorm = seqNorm
		self.epsilon = 1e-8
		self.dropout = torch.nn.Dropout2d(p=0.5, inplace=False) if dropout else None
		self.conv1 = torch.nn.LSTM(dimEncoder, dimEncoder, num_layers=1, batch_first=True)
		self.CharCriterionClassifier = torch.nn.Conv1d(dimEncoder, nPhones + 1, sizeKernel, stride=sizeKernel // 2)
		self.lossCriterion = torch.nn.CTCLoss(blank=nPhones, reduction=reduction, zero_infinity=True)
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

		return self.CharCriterionClassifier(cFeature).permute(0, 2, 1)

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
		loss = self.lossCriterion(predictions, label, featureSize, labelSize).view(1, -1)

		if torch.isinf(loss).sum() > 0 or torch.isnan(loss).sum() > 0:
			loss = 0

		return loss