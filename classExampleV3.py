def __init__(self, src_checkpoint_path=None, dim_size=256, batch_size=8, encoder="audio", cached=True, LSTM=False, freeze=True):
		super().__init__()
		self.dim_size = dim_size
		self.batch_size = batch_size

		#Takes in raw audio/video and return 256 dim outputs
		self.audio_encoder = CPCAudioEncoder(sizeHidden=dim_size)
		self.visual_encoder = CPCVisualEncoder(sizeHidden=dim_size)

		#Take audio and visual final DIMs and return [I need to edit this to add the transformers]
		#Used to return a LSTM output
		self.ar = CPCAudioVisualAR(dim_size, dim_size, False, 1)
		#Applies final convolution
		self.cpc_model = CPCAudioVisualModel(self.audio_encoder, self.visual_encoder, self.ar)
		#Applies LSTM
		self.character_criterion = CTCCharacterCriterion(self.dim_size, 38, LSTM=LSTM)
		#chaches information for fast retrieval
		self.cached = cached

		for g in self.cpc_model.parameters():
			print(g)

		for x in range(10):
			print("")

		if src_checkpoint_path is not None:
			checkpoint = torch.load(src_checkpoint_path)
			self.load_state_dict(checkpoint['state_dict'], strict=False)

		if freeze:
			self.cpc_model.eval()

			for g in self.cpc_model.parameters():
				g.requires_grad = False
				print(g)