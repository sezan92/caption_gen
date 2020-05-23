import cv2
import os
import torch
from torch.utils import data
from torchvision import transforms
from nltk.tokenize import word_tokenize
import random
from PIL import Image
import numpy as np
class Dataset(data.Dataset):
    def __init__(self,images_folder,
                 data,
                 transform,
                 vocab,
                 imsize=(224,224),
                 shuffle=True,
                 tokenizer=word_tokenize
                ):
        self.image_folder=images_folder
        if shuffle:
            random.shuffle(data)
        self.data=data
        self.transform=transform
        self.imsize=imsize
        self.vocab=vocab
        self.tokenizer=tokenizer
    def load_data(self,batch_size=32):
        images,captions=[None]*batch_size,[None]*batch_size
        for i in range(self.__len__()):
            image,caption=self.__getitem__(i)
            c,h,w=image.size()
            images[i%batch_size]=image.view(-1,c,h,w)
            captions[i%batch_size]=caption
            if i>0 and i%batch_size==0:
                yield images,captions
#             if i>=self.__len__():
#                 break
            
    def __getitem__(self,index):
        data =self.data[index]
        image_name=data[0]
        image=cv2.imread(os.path.join(self.image_folder,image_name))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,self.imsize)
        image=self.transform(Image.fromarray(image))
        caption= data[1]
        tokens = self.tokenizer(str(caption).lower())
        caption = []
#         caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()
        return image,caption
        
    def __len__(self):
        return len(self.data)


    