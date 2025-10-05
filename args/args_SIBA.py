class args():
	
	# trainset path
	ir_path = "/home/ws/datasets/image_fusion/MSRS_1083__RoadScene_201/ir"
	vi_path = "/home/ws/datasets/image_fusion/MSRS_1083__RoadScene_201/vi"
	patch_size = 128
	
	model_save_path = "./checkpoint/test"
	use_gpu_number = '1'
	middle_channel = 48

	epochs = 60
	optim_step = 25  
	patch_size = 128

	batch_size = 4 
	init_lr = 1e-4
	use_gpu = True
	optim_gamma = 0.5
	weight_decay = 0