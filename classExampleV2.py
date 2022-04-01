class CPCCharacterClassifierV2(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, sizeHidden=256, visualFeatureDim=512, batch_size=8, numHeads=8, numLayers=6, numLevelsGRU=1, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, encoder="audio", cached=True, LSTM=False, freeze=True):
		super(CPCCharacterClassifier, self).__init__()
		#Set some basic variables (Not sure if this is necessary given that I'm doing it all in one class)
		self.dim_size = dim_size
		self.batch_size = batch_size
		self.DOWNSAMPLING = 160
		normLayer = ChannelNorm
		self.sizeHidden = dim_size
		encoderLayer = nn.TransformerEncoderLayer(d_model=dim_size, nhead=numHeads, dim_feedforward=fcHiddenSize, dropout=dropout)

		#Initialize base audio network
		self.baseAudioNet = nn.Sequential(nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3),
			normLayer(sizeHidden),
			nn.ReLU(),
			nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2),
			normLayer(sizeHidden),
			nn.ReLU(),
			nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1),
			normLayer(sizeHidden),
			nn.ReLU(),
			nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1),
			normLayer(sizeHidden),
			nn.ReLU(),
			nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1),
			normLayer(sizeHidden),
			nn.ReLU())

		#Initialize base video network
		self.baseVideoNet = nn.Sequential(nn.Conv1d(visualFeatureDim, sizeHidden, kernel_size=3, padding=1),
			normLayer(sizeHidden),
			nn.ReLU(),
			nn.ConvTranspose1d(sizeHidden, sizeHidden, kernel_size=4, stride=4),
			normLayer(sizeHidden),
			nn.ReLU())

		#Intialize the LTSM for predictive coding
		self.AR = nn.LSTM(dim_size, dim_size, num_layers=numLevelsGRU, batch_first=True)

		#Declare remaining pre-join network
		self.audioConv = nn.Conv1d(inSize, dim_size, kernel_size=4, stride=4, padding=0)
		self.positionalEncoding = PositionalEncoding(dModel=dim_size, maxLen=peMaxLen)
		self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

		#Declare joint layers
		self.jointNet = nn.Sequential(nn.Conv1d(2*dim_size, dim_size, kernel_size=1, stride=1, padding=0),
			nn.TransformerEncoder(encoderLayer, num_layers=numLayers),
			nn.Conv1d(dim_size, numClasses, kernel_size=1, stride=1, padding=0))

		self.cached = cached

		#Load checkpoints
		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		#Freeze base model
		if freeze:
			self.baseAudioNet.eval()
			self.baseVideoNet.eval()
			self.AR.eval()

			for g in self.baseAudioNet.parameters():
				g.requires_grad = False

			for g in self.baseVideoNet.parameters():
				g.requires_grad = False

			for g in self.AR.parameters():
				g.requires_grad = False

		return

class PositionalEncoding(nn.Module):

	"""
	A layer to add positional encodings to the inputs of a Transformer model.
	Formula:
	PE(pos,2i) = sin(pos/10000^(2i/d_model))
	PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
	"""

	def __init__(self, dModel, maxLen):
		super(PositionalEncoding, self).__init__()
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

class ChannelNorm(nn.Module):

	def __init__(self,
				 numFeatures,
				 epsilon=1e-05,
				 affine=True):

		super(ChannelNorm, self).__init__()
		if affine:
			self.weight = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
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