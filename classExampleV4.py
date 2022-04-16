class FBAudioVisualCPCCharacterClassifierLightningV4(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, batch_size=8, encoder="audio", cached=True, LSTM=False, freeze=True):
		super().__init__()
		self.dim_size = dim_size
		self.batch_size = batch_size

		#Takes in raw audio/video and return 256 dim outputs
		self.audio_encoder = CPCAudioEncoderV4(sizeHidden=dim_size)
		self.visual_encoder = CPCVisualEncoderV4(sizeHidden=dim_size)

		#Take audio and visual final DIMs and return [I need to edit this to add the transformers]
		#Used to return a LSTM output
		self.ar = CPCAudioVisualARV4(dim_size, dim_size, False, 1)
		#Applies final convolution
		self.cpc_model = CPCAudioVisualModelV4(self.audio_encoder, self.visual_encoder, self.ar)
		#Applies LSTM
		self.character_criterion = CTCPhoneCriterionV4(self.dim_size, 38, LSTM=LSTM)
		#chaches information for fast retrieval
		self.cached = cached

		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		if freeze:
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
			cFeature, encodedData, label = self.cpc_model(x, label, padVideo=True, audioVisual=True)
			x_len //= 160
		else:
			cFeature = x

		allLosses = self.character_criterion(cFeature, x_len, label, label_len)

		loss = allLosses.sum()
		return loss

	def get_predictions(self, x):
		cFeature, encodedData, label = self.cpc_model(x, None, padVideo=True, audioVisual=True)
		predictions = torch.nn.functional.softmax(self.character_criterion.getPrediction(cFeature), dim=2)

		return predictions

	def configure_optimizers(self):
		g_params = list(self.character_criterion.parameters())
		optimizer = torch.optim.Adam(g_params, lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
		return optimizer

class CPCAudioEncoderV4(nn.Module):

	def __init__(self,
				 sizeHidden=256):

		super(CPCAudioEncoderV4, self).__init__()
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

class CPCVisualEncoderV4(nn.Module):

	def __init__(self, sizeHidden=256, inputSeqLen=32, visualFeatureDim=512):

		super(CPCVisualEncoderV4, self).__init__()
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

class CPCAudioVisualARV4(nn.Module):

	def __init__(self, dimEncoded, dimOutput, keepHidden, nLevelsGRU):

		super(CPCAudioVisualARV4, self).__init__()
		self.baseNet = nn.LSTM(dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True)
		self.hidden = None
		self.keepHidden = keepHidden

	#Gets DIM size output? pretty sure this was for debugging
	def getDimOutput(self):
		return self.baseNet.hidden_size

	#Feed forward function
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

class CPCAudioVisualModelV4(nn.Module):

	def __init__(self, audioEncoder, visualEncoder, AR):

		super(CPCAudioVisualModelV4, self).__init__()
		self.audioEncoder = audioEncoder
		self.visualEncoder = visualEncoder
		self.gAR = AR
		self.conv0 = nn.Conv1d(self.audioEncoder.sizeHidden, self.audioEncoder.sizeHidden, 1)


	def forward(self, batchData, label, padVideo=False, audioVisual=False):
		audioData, visualData = batchData

		#encode audio
		encodedAudio = self.audioEncoder(audioData)

		#encode video
		encodedVideo = self.visualEncoder(visualData)

		if padVideo:
			encodedVideo = F.pad(encodedVideo, (0, encodedAudio.shape[2]-encodedVideo.shape[2]))

		#merge encodings, conv, and permute
		encodedAudioVisual = F.relu(self.conv0(encodedAudio+encodedVideo))
		encodedAudioVisual = encodedAudioVisual.permute(0, 2, 1)

		#permute audio only features
		encodedAudio = encodedAudio.permute(0, 2, 1)

		#run context AR network
		cFeature = self.gAR(encodedAudioVisual)

		if not audioVisual:
			return cFeature, encodedAudio, label
		else:
			return cFeature, encodedAudioVisual, label

class CTCPhoneCriterionV4(torch.nn.Module):

	def __init__(self, dimEncoder, nPhones, LSTM=False, sizeKernel=8,
				 seqNorm=False, dropout=False, reduction='mean'):

		super(CTCPhoneCriterionV4, self).__init__()
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