import torch
import urllib
from PIL import Image
from torchvision import transforms
import sys
import time

from googlenet import googlenet
from check import check

model = googlenet(pretrained=True)
model.eval()

each_accuracy = 0
accuracy = 0

else_animal = [0, 0, 0] #leopard, jaguar, cheetah

start = time.time()

for i in range(100):
    input_image = Image.open(f'../../data/n02128385/{i}.JPEG')
    Presume_animal = check(input_image, model)
    if Presume_animal == "leopard":
        each_accuracy += 1
        accuracy += 1
    elif Presume_animal == "jaguar":
        else_animal[1] += 1
    elif Presume_animal == "cheetah":
        else_animal[2] += 1
    
print("leopard_finish :", time.time() - start)
print(f'leopard accuracy : {each_accuracy / 100 * 100 }')
print(f'else : {else_animal[1]}, {else_animal[2]}')
each_accuracy = 0
else_animal = [0, 0, 0]
start = time.time()

for i in range(100):
    input_image = Image.open(f'../../data/n02128925/{i}.JPEG')
    Presume_animal = check(input_image, model)
    if Presume_animal == "jaguar":
        accuracy += 1
        each_accuracy += 1
    elif Presume_animal == "leopard":
        else_animal[0] += 1
    elif Presume_animal == "cheetah":
        else_animal[2] += 1


print("jaguar finish :", time.time() - start)
print(f'jaguar accuracy : {each_accuracy / 100 * 100 }')
print(f'else : {else_animal[0]}, {else_animal[2]}')
each_accuracy = 0
start = time.time()
else_animal = [0, 0, 0]

for i in range(100):
    input_image = Image.open(f'../../data/n02130308/{i}.JPEG')
    Presume_animal = check(input_image, model)
    if Presume_animal == "cheetah":
        accuracy += 1
        each_accuracy += 1
    elif Presume_animal == "leopard":
        else_animal[0] += 1
    elif Presume_animal == "jaguar":
        else_animal[1] += 1

print("cheetah finish :", time.time() - start)
print(f'cheetah accuracy : {each_accuracy / 100 * 100 }')
print(f'else : {else_animal[0]}, {else_animal[1]}')


print(f'percent = {accuracy / 300 * 100}')