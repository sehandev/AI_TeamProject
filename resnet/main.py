import os
import urllib

import torch
from PIL import Image
from torchvision import transforms

# Custom
from resnet import _resnet
from helper import get_class_id, save_model


def open_image(class_name, index):
  class_id = get_class_id(class_name)
  file_name = f'{class_id}_{index}.JPEG'
  file_path = os.path.join("data", class_id, file_name)

  input_image = Image.open(file_path)

  return input_image

def get_preprocess_function():
  preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])

  return preprocess

def debug_model(model):
  for m in model.modules():
      print(m)

def main(model_name, is_pretrained, class_name, index):

  # Init model
  model = _resnet(model_name, 1000, is_pretrained)

  # TEST
  # return debug_model(model)

  # Save finetuned model
  # save_model(model_name, model)

  # Load image
  input_image = open_image(class_name, index)

  preprocess = get_preprocess_function()
  input_tensor = preprocess(input_image)  # [3, 224, 224]
  input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]
  
  # GPU
  if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

  with torch.no_grad():
    output = model(input_batch)  # [num_class]

  # Calculate probability_array
  probability_array = torch.nn.functional.softmax(output, dim=0)  # [num_class]

  # Read the categorie_list
  with open("data/imagenet_classes.txt", "r") as f:
    categorie_list = [s.strip() for s in f.readlines()]
    
  # Select top k from probability array
  K = 3
  print(f'\n [ {model_name} Top {K} ]')
  topk_array, topk_category_index = torch.topk(probability_array, K)
  for i in range(len(topk_array)):
    class_name = categorie_list[topk_category_index[i]]
    probability = topk_array[i].item() * 100
    print(f'{class_name:<10} : {probability:6.3f}%')


if __name__ == "__main__":
  class_name = 'leopard'
  index = 17

  print(f' [ Predict {class_name} - {index} ]')
  main('resnet50', True, class_name, index)
  main('resnet101', True, class_name, index)
  main('resnet152', True, class_name, index)
