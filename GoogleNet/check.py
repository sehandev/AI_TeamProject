import torch
import urllib
from PIL import Image
from torchvision import transforms

from googlenet import googlenet

def check(input_image, model):

	preprocess = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize((0.485,), (0.225,)),
	])

	input_tensor = preprocess(input_image)

	# gray scale image to rgb
	if len(input_tensor) == 1:
		input_tensor = torch.cat((input_tensor, input_tensor, input_tensor), dim=0)

	input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

	# move the input and model to GPU for speed if available
	if torch.cuda.is_available():
	    input_batch = input_batch.to('cuda')
	    model.to('cuda')

	output = model(input_batch)

	# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
	#print(output[0])
	# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
	probabilities = torch.nn.functional.softmax(output[0], dim=0)

	CLASS_IDS = {
		'leopard' : 'n02128385',
		'jaguar' : 'n02128925',
		'cheetah' : 'n02130308',
	}

	#CLASS_NAME_LIST = list(CLASS_IDS.keys())
	#CLASS_ID_LIST = list(CLASS_IDS.values())

	with open("imagenet_classes.txt", "r") as f:
		categories = [s.strip() for s in f.readlines()]

	# Show top categories per image
	top3_prob, top3_catid = torch.topk(probabilities, 1)
	#for i in range(top3_prob.size(0)):
	#	print(categories[top3_catid[i]], top3_prob[i].item())
	return categories[top3_catid[0]]
