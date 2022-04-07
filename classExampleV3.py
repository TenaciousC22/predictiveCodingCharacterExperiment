class CPCCharacterClassifierV3(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, batch_size=8, visualFeatureDim=512, numHeads=8, numLayers=6, numLevelsGRU=1, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, encoder="audio", cached=True, LSTM=False, freeze=True):
		super(CPCCharacterClassifierV3, self).__init__()
		self.dim_size = dim_size
		self.batch_size = batch_size

		#Takes in raw audio/video and return 256 dim outputs
		self.audioFront = CPCAudioEncoderV2(sizeHidden=dim_size)
		self.visualFront = CPCVisualEncoderV2(sizeHidden=dim_size)

		self.ar = CPCAudioVisualARV2(dim_size, dim_size, keepHidden=False)

		#Create Unified Model
		self.cpc_model = CPCAudioVisualModelV2(self.audioFront, self.visualFront, self.ar)

		#Applies LSTM
		self.character_criterion = CTCCharacterCriterion(self.dim_size, 38, LSTM=LSTM)

		self.cached=cached

		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		for g in self.audioFront.parameters():
			print(g)

		sleep(2)

		for g in self.visualFront.parameters():
			print(g)

		sleep(2)

		if freeze:
			self.audioFront.eval()
			self.visualFront.eval()

			for g in self.audioFront.parameters():
				g.requires_grad = False
				print(g)

			sleep(2)

			for g in self.visualFront.parameters():
				g.requires_grad = False
				print(g)

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
			cFeature, encodedData, label = self.cpc_model(x, label, padVideo=True)
			x_len //= 160
		else:
			cFeature = x

		allLosses = self.character_criterion(cFeature, x_len, label, label_len)

		loss = allLosses.sum()
		return loss

	def get_predictions(self, x):
		cFeature, decodedData, label = self.cpc_model(x, None, padVideo=True)
		predictions = torch.nn.functional.softmax(self.character_criterion.getPrediction(cFeature), dim=2)

		return predictions

	def configure_optimizers(self):
		g_params = list(self.character_criterion.parameters())
		optimizer = torch.optim.Adam(g_params, lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
		return optimizer

class CPCAudioVisualModelV2(nn.Module):

	def __init__(self, audioEncoder, visualEncoder, AR):

		super(CPCAudioVisualModelV2, self).__init__()
		self.audioEncoder = audioEncoder
		self.visualEncoder = visualEncoder
		self.gAR = AR
		#self.conv0 = nn.Conv1d(self.audioEncoder.sizeHidden, self.audioEncoder.sizeHidden, 1)


	def forward(self, batchData, label, padVideo=False):
		audioData, visualData = batchData

		#encode audio
		encodedAudio = self.audioEncoder(audioData)

		#encode video
		encodedVideo = self.visualEncoder(visualData)

		if padVideo:
			encodedVideo = F.pad(encodedVideo, (0, encodedAudio.shape[2]-encodedVideo.shape[2]))

		#run context AR network
		jointDecoded = self.gAR(encodedAudio,encodedVideo)

		return cFeature, jointDecoded, label

class CPCAudioVisualARV2(nn.Module):

	def __init__(self, inSize=256, dim_size=256, peMaxLen=2500, keepHidden=False, numHeads=8, numLayers=6, fcHiddenSize=2048, dropout=0.1, numClasses=39, cached=True):

		super(CPCAudioVisualARV2, self).__init__()
		#self.baseNet = nn.LSTM(dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True)
		self.hidden = None
		self.keepHidden = keepHidden
		encoderLayer = nn.TransformerEncoderLayer(d_model=dim_size, nhead=numHeads, dim_feedforward=fcHiddenSize, dropout=dropout)

		#Declare remaining pre-join network
		self.audioConv = nn.Conv1d(inSize, dim_size, kernel_size=4, stride=4, padding=0)
		self.positionalEncoding = PositionalEncodingV2(dModel=dim_size, maxLen=peMaxLen)
		self.audioEncoder= nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

		#Declare joint layers
		self.jointConv = nn.Conv1d(2*dim_size, dim_size, kernel_size=1, stride=1, padding=0)
		self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		#self.outputConv = nn.Conv1d(dim_size, numClasses, kernel_size=1, stride=1, padding=0)

		self.cached = cached

	#Gets DIM size output? pretty sure this was for debugging
	# def getDimOutput(self):
	# 	return self.baseNet.hidden_size

	#Feed forward function
	def forward(self, encodedAudio, encodedVisual):

		#Not sure if the transpose functions are necessary
		encodedAudio = encodedAudio.transpose(0, 1).transpose(1, 2)
		audioBatch = self.audioConv(encodedAudio)
		audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
		audioBatch = self.positionalEncoding(audioBatch)
		audioBatch = self.audioEncoder(audioBatch)

		videoBatch = self.positionalEncoding(encodedVideo)
		videoBatch = self.videoEncoder(encodedVideo)

		jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
		jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
		jointBatch = self.jointConv(jointBatch)
		jointBatch=jointBatch.transpose(1, 2).transpose(0, 1)

		jointBatch = slef.jointDecoder(jointBatch)
		output = jointBatch.transpose(0, 1).transpose(1, 2)
		# jointBatch = self.outputConv(jointBatch)
		# jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
		# outputBatch = F.log_softmax(jointBatch, dim=2)

		return outputBatch

class CTCCharacterCriterionV2(torch.nn.Module):

	def __init__(self, dimEncoder, nPhones, LSTM=False, sizeKernel=8,
				 seqNorm=False, dropout=False, reduction='mean'):

		super(CTCCharacterCriterionV2, self).__init__()
		self.seqNorm = seqNorm
		self.epsilon = 1e-8
		self.dropout = torch.nn.Dropout2d(p=0.5, inplace=False) if dropout else None
		self.conv1 = torch.nn.LSTM(dimEncoder, dimEncoder, num_layers=1, batch_first=True)
		self.PhoneCriterionClassifier = torch.nn.Conv1d(dimEncoder, nPhones + 1, sizeKernel, stride=sizeKernel // 2)
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

class CPCAudioEncoderV2(nn.Module):

	def __init__(self,
				 sizeHidden=256):

		super(CPCAudioEncoderV2, self).__init__()
		normLayer = ChannelNorm
		self.sizeHidden = sizeHidden
		self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
		self.batchNorm0 = normLayer(sizeHidden)
		self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
		self.batchNorm1 = normLayer(sizeHidden)
		self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
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

class CPCVisualEncoderV2(nn.Module):

	def __init__(self, sizeHidden=256, inputSeqLen=32, visualFeatureDim=512):

		super(CPCVisualEncoderV2, self).__init__()
		normLayer = ChannelNorm
		self.inputSeqLen = inputSeqLen
		self.sizeHidden = sizeHidden

		self.conv0 = nn.Conv1d(visualFeatureDim, sizeHidden, kernel_size=3, padding=1)
		self.batchNorm0 = normLayer(sizeHidden)

		self.conv1 = nn.ConvTranspose1d(sizeHidden, sizeHidden, kernel_size=4, stride=4)
		self.batchNorm1 = normLayer(sizeHidden)


	def getDimOutput(self):
		return self.conv0.out_channels

	def forward(self, x):
		x = F.relu(self.batchNorm0(self.conv0(x)))
		x = F.relu(self.batchNorm1(self.conv1(x)))
		return x

class PositionalEncodingV2(nn.Module):

	"""
	A layer to add positional encodings to the inputs of a Transformer model.
	Formula:
	PE(pos,2i) = sin(pos/10000^(2i/d_model))
	PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
	"""

	def __init__(self, dModel, maxLen):
		super(PositionalEncodingV2, self).__init__()
		pe = torch.zeros(maxLen, dModel)
		position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
		denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
		pe[:, 0::2] = torch.sin(position/denominator)
		pe[:, 1::2] = torch.cos(position/denominator)
		pe = pe.unsqueeze(dim=0).transpose(0, 1)
		self.register_buffer("pe", pe)


	def forward(self, inputBatch):
		outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
		return outputBatch