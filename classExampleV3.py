class CPCCharacterClassifierV3(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, batch_size=8, visualFeatureDim=512, numHeads=8, numLayers=6, numLevelsGRU=1, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, encoder="audio", cached=True, LSTM=False, freeze=True):
		super(CPCCharacterClassifierV3, self).__init__()
		self.dim_size = dim_size
		self.batch_size = batch_size

		#Takes in raw audio/video and return 256 dim outputs
		self.audioFront = CPCAudioEncoder(sizeHidden=dim_size)
		self.visualFront = CPCVisualEncoder(sizeHidden=dim_size)

		self.ar = CPCAudioVisualARV2(dim_size, dim_size, False, 1)

		#Create Unified Model
		self.cpc_model = CPCAudioVisualModelV2(self.audioFront, self.visualFront)

		#Applies LSTM
		#self.character_criterion = CTCCharacterCriterion(self.dim_size, 38, LSTM=LSTM)

		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		if freeze:
			self.audioFront.eval()
			self.visualFront.eval()

			for g in self.audioFront.parameters():
				g.requires_grad = False

			for g in self.visualFront.parameters():
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
		predictions = torch.nn.functional.softmax(self.phoneme_criterion.getPrediction(cFeature), dim=2)

		return predictions

	def configure_optimizers(self):
		g_params = list(self.phoneme_criterion.parameters())
		optimizer = torch.optim.Adam(g_params, lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
		return optimizer

class CPCAudioVisualModelV2(nn.Module):

	def __init__(self, audioEncoder, visualEncoder, AR):

		super(CPCAudioVisualModelV2, self).__init__()
		self.audioEncoder = audioEncoder
		self.visualEncoder = visualEncoder
		self.gAR = AR
		#self.conv0 = nn.Conv1d(self.audioEncoder.sizeHidden, self.audioEncoder.sizeHidden, 1)


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

class CPCAudioVisualARV2(nn.Module):

	def __init__(self, inSize=256, dim_size=256, peMaxLen=2500, keepHidden=False, numHeads=8, numLayers=6, fcHiddenSize=2048, dropout=0.1, numClasses=39):

		super(CPCAudioVisualARV2, self).__init__()
		#self.baseNet = nn.LSTM(dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True)
		self.hidden = None
		self.keepHidden = keepHidden
		encoderLayer = nn.TransformerEncoderLayer(d_model=dim_size, nhead=numHeads, dim_feedforward=fcHiddenSize, dropout=dropout)

		#Declare remaining pre-join network
		self.audioConv = nn.Conv1d(inSize, dim_size, kernel_size=4, stride=4, padding=0)
		self.positionalEncoding = PositionalEncoding(dModel=dim_size, maxLen=peMaxLen)
		self.audioEncoder= nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

		#Declare joint layers
		self.jointConv = nn.Conv1d(2*dim_size, dim_size, kernel_size=1, stride=1, padding=0)
		self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.outputConv = nn.Conv1d(dim_size, numClasses, kernel_size=1, stride=1, padding=0)

		self.cached = cached

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