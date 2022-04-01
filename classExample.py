class CPCCharacterClassifier(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, sizeHidden=256, visualFeatureDim=512, batch_size=8, numHeads=8, numLayers=6, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, encoder="audio", cached=True, LSTM=False, freeze=True):
		super(CPCCharacterClassifier, self).__init__()
		#Set some basic variables (Not sure if this is necessary given that I'm doing it all in one class)
		self.dim_size = dim_size
		self.batch_size = batch_size
		self.DOWNSAMPLING = 160
		normLayer = ChannelNorm
		self.sizeHidden = dim_size

		#Initialize base network
		self.baseNet=CPCBaseNetwork()

		#Declare remaining network
		self.audioConv = nn.Conv1d(inSize, dim_size, kernel_size=4, stride=4, padding=0)
		self.positionalEncoding = PositionalEncoding(dModel=dim_size, maxLen=peMaxLen)
		encoderLayer = nn.TransformerEncoderLayer(d_model=dim_size, nhead=numHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
		self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.jointConv = nn.Conv1d(2*dim_size, dim_size, kernel_size=1, stride=1, padding=0)
		self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.outputConv = nn.Conv1d(dim_size, numClasses, kernel_size=1, stride=1, padding=0)


		#Load checkpoints
		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		#Freeze base model
		if freeze:
			self.baseNet.eval()

			for g in self.baseNet.parameters():
				g.requires_grad = False

		return

class CPCBaseNetwork(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, sizeHidden=256, visualFeatureDim=512, batch_size=8, numHeads=8, numLayers=6, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, encoder="audio", cached=True, LSTM=False, freeze=True):
		super(CPCBaseNetwork, self).__init__()
		#Set some basic variables (Not sure if this is necessary given that I'm doing it all in one class)
		self.dim_size = dim_size
		self.batch_size = batch_size
		self.DOWNSAMPLING = 160
		normLayer = ChannelNorm
		self.sizeHidden = dim_size

		#Declare audio base network (basically just copied from CPCAudioEncoder)
		self.audioConv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
		self.audioBatchNorm0 = normLayer(sizeHidden)
		self.audioConv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
		self.audioBatchNorm1 = normLayer(sizeHidden)
		self.audioConv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
		self.audioBatchNorm2 = normLayer(sizeHidden)
		self.audioConv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
		self.audioBatchNorm3 = normLayer(sizeHidden)
		self.audioConv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
		self.audioBatchNorm4 = normLayer(sizeHidden)

		#Declare video base network (basically just copied from CPCVideoEncoder)
		self.videoConv0 = nn.Conv1d(visualFeatureDim, sizeHidden, kernel_size=3, padding=1)
		self.videoBatchNorm0 = normLayer(sizeHidden)
		self.videoConv1 = nn.ConvTranspose1d(sizeHidden, sizeHidden, kernel_size=4, stride=4)
		self.videoBatchNorm1 = normLayer(sizeHidden)

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