class CPCCharacterClassifierV3(pl.LightningModule):
	def __init__(self, src_checkpoint_path=None, dim_size=256, batch_size=8, visualFeatureDim=512, numHeads=8, numLayers=6, numLevelsGRU=1, peMaxLen=2500, inSize=256,
			fcHiddenSize=2048, dropout=0.1, numClasses=38, encoder="audio", cached=True, LSTM=False, freeze=True):
		super(CPCCharacterClassifierV3, self).__init__()
		self.dim_size = dim_size
		self.batch_size = batch_size
		encoderLayer = nn.TransformerEncoderLayer(d_model=dim_size, nhead=numHeads, dim_feedforward=fcHiddenSize, dropout=dropout)

		#Takes in raw audio/video and return 256 dim outputs
		self.audio_encoder = CPCAudioEncoder(sizeHidden=dim_size)
		self.visual_encoder = CPCVisualEncoder(sizeHidden=dim_size)

		#Take audio and visual final DIMs and return [I need to edit this to add the transformers]
		#Used to return a LSTM output
		# self.ar = CPCAudioVisualAR(dim_size, dim_size, False, 1)
		# #Applies final convolution
		# self.cpc_model = CPCAudioVisualModel(self.audio_encoder, self.visual_encoder, self.ar)
		#Applies LSTM
		#self.character_criterion = CTCCharacterCriterion(self.dim_size, 38, LSTM=LSTM)
		#chaches information for fast retrieval

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

		for g in self.audio_encoder.parameters():
			print(g)

		for x in range(10):
			print("")

		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		if freeze:
			self.audio_encoder.eval()
			self.visual_encoder.eval()

			for g in self.audio_encoder.parameters():
				g.requires_grad = False
				print(g)

			for g in self.visual_encoder.parameters():
				g.requires_grad = False