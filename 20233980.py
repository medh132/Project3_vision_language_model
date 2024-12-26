import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import numpy as np
from PIL import Image
import json
from datasets import load_dataset
import copy
import os
import torch.nn.functional as F
import argparse
import random
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchvision.models import densenet121



# Constants
CIFAR_BATCH_SIZE = 128
LM_BATCH_SIZE = 32
VL_BATCH_SIZE = 16
MAX_LENGTH = 128
HIDDEN_SIZE = 768
NUM_EPOCHS = 1
IMG_PATCH = '<img>'
NUM_IMG_TOKEN = 32
VLM_MAX_LENGTH = 32

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CIFAR-10 Dataset and DataLoader
def get_cifar10_loaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=CIFAR_BATCH_SIZE,
                           shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=CIFAR_BATCH_SIZE,
                          shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ELI5 Dataset
class ELI5Dataset(Dataset):
    def __init__(self,tokenizer, MAX_POSITION_EMBEDDINGS, data_type):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.block_size = MAX_POSITION_EMBEDDINGS
        
        if data_type == "train":
            data = load_dataset("eli5_category", split="train[:3000]", trust_remote_code=True)
            data = data.select(range(1000))
        elif data_type == "valid":
            data = load_dataset("eli5_category", split="validation1[:2000]", trust_remote_code=True)
        elif data_type == "test":
            data = load_dataset("eli5_category", split="test[:20]", trust_remote_code=True)

        data = data.flatten() 
        data = data.map(self.preprocess_function, batched=True,num_proc=8,remove_columns=data.column_names)
        data = data.map(self.group_texts, batched=True, num_proc=8)
        result =[]
        for i in data:
            result.append(i['input_ids'])
        self.final_data = torch.tensor(result).to(torch.int64)
        
    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])
    
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) 
        if total_length >= (self.block_size-2):
            total_length = (total_length // (self.block_size-2)) * (self.block_size-2)
        result = {
            k: [[self.tokenizer.bos_token_id]+t[i : i + self.block_size-2]+[self.tokenizer.eos_token_id] for i in range(0, total_length, self.block_size-2)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx):
        return self.final_data[idx]

# LLaVA Dataset
def transform_fn(is_train):
    if is_train:
        return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    else:
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Do not change
class LLaVADataset(Dataset):
    def __init__(self, json_file, img_path, tokenizer, is_train):
        super().__init__()

        self.transform = transform_fn(is_train)

        self.json_file = json_file

        self.tokenizer = tokenizer
        self.img_path = img_path

        self.ignore_idx = -100
        self.begin_signal = tokenizer.bos_token
        self.end_signal = tokenizer.eos_token

        with open(self.json_file) as json_file:
            data = json.load(json_file)

        if is_train:
            data = data[:1000]
        else:
            data = data[1000:]

        self.data = data

    def preprocess(self, conversation):
        question = self.begin_signal + "human: " + conversation[0]['value'] + self.end_signal
        answer = self.begin_signal + "assistant: " + conversation[1]['value'] + self.end_signal

        tokenized_q = self.tokenizer(question, return_tensors="pt")

        combined_qa = question + answer
        tokenized_qa = self.tokenizer(combined_qa, padding="max_length", truncation=True,
                                      max_length=VLM_MAX_LENGTH, return_tensors="pt")

        input_ids = tokenized_qa.input_ids[0]
        label = copy.deepcopy(input_ids)
        len_of_q = len(tokenized_q.input_ids[0])
        label[:len_of_q] = self.ignore_idx

        len_of_pad = tokenized_qa.input_ids.eq(self.tokenizer.pad_token_id).sum().item()
        label[-len_of_pad:] = self.ignore_idx

        return input_ids, label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta = self.data[idx]

        image_id = meta['image']
        image = Image.open(os.path.join(self.img_path, image_id)).convert('RGB')
        image = self.transform(image)

        conversation = meta['conversation']
        input_id, label = self.preprocess(conversation)

        return dict(image=image, input_ids=input_id, label=label)


# Hyperparameters
CIFAR_EPOCHS = 50 #5
VLM_EPOCHS = 50 #5
ELI5_EPOCHS = 30 #3
LEARNING_RATE = 5e-5
ELI5_LEARNING_RATE = 1e-5

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        return self.relu(x)

class VisionLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, tokenizer):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = nn.Sequential(*list(vision_encoder.children())[:-1])
        self.resnet_block = ResNetBlock(1024, text_decoder.config.hidden_size)
        self.text_decoder = text_decoder
        self.tokenizer = tokenizer

    def forward(self, images, input_ids, labels=None):
        vision_feats = self.vision_encoder(images)
        vision_feats = vision_feats.mean(dim=[2, 3])
        proj_feats = self.resnet_block(vision_feats.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        batch_size = images.size(0)
        img_tokens = proj_feats.unsqueeze(1).expand(-1, NUM_IMG_TOKEN, -1)
        
        input_embeds = torch.cat((img_tokens, self.text_decoder.transformer.wte(input_ids)), dim=1)
        
        if labels is not None:
            # Create ignore labels (-100) for image tokens
            img_labels = torch.full((batch_size, NUM_IMG_TOKEN), -100, device=labels.device)
            # Concatenate with original labels
            labels = torch.cat((img_labels, labels), dim=1)
        
        outputs = self.text_decoder(inputs_embeds=input_embeds, labels=labels)
        return outputs.loss, outputs.logits

def train_vision_encoder(train_loader, device, num_epochs=CIFAR_EPOCHS):
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Training vision encoder...")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Vision Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    
    return model

def train_text_decoder(text_decoder, train_loader, device, num_epochs=ELI5_EPOCHS):
    optimizer = AdamW(text_decoder.parameters(), lr=ELI5_LEARNING_RATE)
    text_decoder.train()
    
    print("Training text decoder...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = text_decoder(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"ELI5 Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    return text_decoder

def train_vlm(model, train_loader, device, num_epochs=VLM_EPOCHS):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    print("Training vision-language model...")
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            loss, _ = model(images, input_ids, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 9:
                print(f'VLM Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss/10:.4f}')
                total_loss = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                      help='train: train model and generate logits, inference: load model and generate logits')
    parser.add_argument('--model_path', type=str, default='vlm_model.pth',
                      help='Path to load/save model weights')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens('<IMG>', special_tokens=True)

    # Create test dataloader (needed for both modes)
    test_llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=False)
    test_llava_loader = DataLoader(test_llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=False)

    if args.mode == 'train':
        # Get training dataloaders
        cifar_trainloader, _ = get_cifar10_loaders()
        eli5_dataset = ELI5Dataset(tokenizer, MAX_LENGTH, 'train')
        eli5_loader = DataLoader(eli5_dataset, batch_size=LM_BATCH_SIZE, shuffle=True)
        llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=True)
        llava_loader = DataLoader(llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=True)

        # Train vision encoder
        vision_encoder = train_vision_encoder(cifar_trainloader, device)
        
        # Train text decoder
        text_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        text_decoder.resize_token_embeddings(len(tokenizer))
        text_decoder = text_decoder.to(device)
        text_decoder = train_text_decoder(text_decoder, eli5_loader, device)
        
        # Train VLM
        vlm_model = VisionLanguageModel(vision_encoder, text_decoder, tokenizer).to(device)
        train_vlm(vlm_model, llava_loader, device)
        
        # Save model
        print("Saving VLM model...")
        torch.save({
            'vision_encoder_state_dict': vlm_model.vision_encoder.state_dict(),
            'resnet_block_state_dict': vlm_model.resnet_block.state_dict(),
            'text_decoder_state_dict': vlm_model.text_decoder.state_dict(),
        }, args.model_path)
        print(f"Model saved to {args.model_path}")

    else:  # inference mode
        print(f"Loading model from {args.model_path}")
        # Initialize base models
        vision_encoder = densenet121(pretrained=True)
        vision_encoder.classifier = nn.Linear(vision_encoder.classifier.in_features, 10)
        text_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        text_decoder.resize_token_embeddings(len(tokenizer))
        
        # Create and load VLM model
        vlm_model = VisionLanguageModel(vision_encoder, text_decoder, tokenizer).to(device)
        checkpoint = torch.load(args.model_path)
        vlm_model.vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'])
        vlm_model.resnet_block.load_state_dict(checkpoint['resnet_block_state_dict'])
        vlm_model.text_decoder.load_state_dict(checkpoint['text_decoder_state_dict'])

    # Generate test logits (common for both modes)
    print("Generating test set logits...")
    vlm_model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in test_llava_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            _, logits = vlm_model(images, input_ids)
            # Remove the image token positions and last vocab dimension
            logits = logits[:, NUM_IMG_TOKEN:, :50257]  
            all_logits.append(logits.cpu().numpy())

    # Save logits
    logits_file = '20233980.npy'
    logits_array = np.concatenate(all_logits, axis=0)
    print(f"Logits shape: {logits_array.shape}")  # Should now be (20, 32, 50257)
    np.save(logits_file, logits_array)
    print(f"Saved logits to {logits_file}")


if __name__ == "__main__":
    main()

