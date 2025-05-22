# -*- coding: utf-8 -*-

from datasets import load_dataset, DatasetDict,  concatenate_datasets
import torch
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


""" 
Data prepping:
acquiring data, exploring dataset, cleaning / wrangling data, splitting data into training and validation set, tokenizing data
"""
ciphers = ['Rot13', 'Atbash', 'Polybius', 'Vigenere', 'Reverse', 'SwapPairs', 'ParityShift', 'DualAvgCode', 'WordShift']

# Download dataset from HF. As there are mupltiple subsets, combine them into one for training
datasets = []
for cipher in ciphers:
    ds = load_dataset("yu0226/CipherBank", cipher, split="test")
    ds = ds.map(lambda x: {'cipher_type': cipher})
    datasets.append(ds)

dataset = concatenate_datasets(datasets)

#splitting dataset
dataset = dataset.shuffle(seed=12)
split_dataset = dataset.train_test_split(test_size=0.2)

dataset = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

#print(dataset.keys())
print("\n")
# print a sample from the dataset
print("Train sample: ", dataset["train"][:3])
print("\n")
print("Test sample: ", dataset["test"][:3])

#load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def massage_input_text(example):
    """
    Helper function for converting inputs into single string
    :param example: Sample input from the dataset
    :return: Formatted training string containing question and text
    """
    cipher_type = example["cipher_type"]
    ciphertext = example[cipher_type]

    input_text = f"Identify the cipher used in the ciphertext:\n{ciphertext}"

    return {"text": input_text}

def add_label(example):
    """
    Helper function for adding correct label
    :param example: Sample input from the dataset
    :return: Formatted training string containing correct label
    """
    return {"labels": ciphers.index(example["cipher_type"])}

# process input of train and validation sets
massaged_datasets = dataset.map(massage_input_text)
massaged_datasets = massaged_datasets.map(add_label)

# massaged_datasets["train"][0]

def tokenize(tokenizer, example):
    """
    Helper for pre-tokenizing all examples.
    """
    tokenized = tokenizer(
        example["text"],
        #fixing length to avoid colab memory issue
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized

tokenized_dataset = massaged_datasets.map(
    lambda example: tokenize(tokenizer, example),
    batched=True,
    remove_columns= ["domain", "sub_domain", "plaintext", 'cipher_type', 'Rot13', 'Atbash', 'Polybius', 'Vigenere', 'Reverse', 'SwapPairs', 'ParityShift', 'DualAvgCode', 'WordShift']
)

# move to accelerated device if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: {device}")
else:
    device = torch.device("cpu")
    print(f"Device: {device}")

# load pretrained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=9).to(device)

# instantiate tokenized train dataset
train_dataset = tokenized_dataset["train"]
# instantiate tokenized validation dataset
validation_dataset = tokenized_dataset["test"]

# instantiate a data collator
collate_fn = default_data_collator

# create a DataLoader for the dataset to automatically batch the data
# and iteratively return training examples in batches
dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
)

# create a DataLoader for the test dataset
# reason for separate data loader is
# different index for retreiving the test batches and different batch size
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# put the model in training mode
model.train()

model = model.to(device)

# training configutations
epochs = 5
train_steps = len(dataloader)
print("Number of training steps: ", train_steps)
num_test_steps = 150
# define optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=0.01)
# define for losses accumulation
losses = []
validation_losses = []

# get a batch of data
dataloader_iter = iter(dataloader)

# iterate over epochs
for e in range(epochs):
    # iterate over training steps
    for i in tqdm(range(train_steps)):
        x = next(dataloader_iter)
        # move the data to the device
        x = {k: v.to(device) for k, v in x.items()}

        # forward pass through the model
        outputs = model(**x)
        # get the loss
        loss = outputs.loss
        # backward pass
        loss.backward()
        # update the parameters of the model
        losses.append(loss.item())
        optimizer.step()
        # zero out gradient for next step
        optimizer.zero_grad()

        # evaluate on a few steps of validation set
        if i % 10 == 0:
            print(f"Epoch {e}, step {i}, loss {loss.item()}")
            val_loss = 0
            for j in range(num_test_steps):
                x_test = next(iter(validation_dataloader))
                x_test = {k: v.to(device) for k, v in x_test.items()}
                with torch.no_grad():
                    test_outputs = model(**x_test)
                val_loss += test_outputs.loss.item()
            validation_losses.append(val_loss / num_test_steps)
            print("Test loss: ", val_loss / num_test_steps)

# plot training losses on x axis
plt.plot(losses, label="Training Loss")
plt.plot(
    [i * 10 for i in range(len(validation_losses))],
    label="Validation Loss"
)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

plt.plot( losses, label="Loss train")
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot( validation_losses, label="Loss train")
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend()
plt.show()


# construct a list of questions without the ground truth label
# and compare prediction of the model with the ground truth
def construct_test_samples(example):
    """
    Helper for converting input examples into a single string for testing the model.
    example: dict
        Sample input from the dataset
    input_text: str, str
        Tuple: Formatted test text which contains the ciphertext and label
    """
    cipher_type = example["cipher_type"]
    ciphertext = example[cipher_type]

    input_text = f"Identify the cipher used in the ciphertext:\n{ciphertext}"

    return {"text": input_text, "label": cipher_type}

test_samples = [construct_test_samples(dataset["test"][i]) for i in range(10)]
#test_samples

label_map = {
    "LABEL_0": "Rot13",
    "LABEL_1": "Atbash",
    "LABEL_2": "Polybius",
    "LABEL_3": "Vigenere",
    "LABEL_4": "Reverse",
    "LABEL_5": "SwapPairs",
    "LABEL_6": "ParityShift",
    "LABEL_7": "DualAvgCode",
    "LABEL_8": "WordShift",
}
# Test the model

# set it to evaluation mode
model.eval()

predictions = []
for sample in test_samples:
    input_text = sample["text"]
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_id = logits.argmax(dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]

    predictions.append((input_text, predicted_label, sample["label"]))

print("Predictions of trained model:")
for i, (input_text, predicted_token, true_label) in enumerate(predictions, 1):
    predicted_label = label_map.get(predicted_token, predicted_token)
    print(f"Example {i}:")
    print(f"Input:{input_text}")
    print(f"Predicted Label: {predicted_label}")
    print(f"True Label: {true_label}")