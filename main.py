# -*- coding: utf-8 -*-

from datasets import load_dataset, DatasetDict,  concatenate_datasets
import torch
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


ciphers = ['Rot13', 'Atbash', 'Polybius', 'Vigenere', 'Reverse', 'SwapPairs', 'ParityShift', 'DualAvgCode', 'WordShift']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset():
    """
    Download dataset from HF. As there are mupltiple subsets, combine them into one for training
    """
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

    return dataset


def massage_datasets(example):
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

def process_dataset(dataset, tokenizer):
    dataset = dataset.map(massage_datasets)
    dataset = dataset.map(add_label)

    dataset = dataset.map(
        lambda example: tokenize(tokenizer, example),
        batched=True,
        remove_columns=["domain", "sub_domain", "plaintext", "cipher_type"] + ciphers
    )
    return dataset


def load_model():
    """
    load pretrained BERT model
    """
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=9)
    return model.to(device)


def train(model, train_loader, val_loader, train_steps, epochs=5, val_steps=150):
    # define optimizer and learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # define for losses accumulation
    losses = []
    validation_losses = []

    # get a batch of data
    dataloader_iter = iter(train_loader)
    # put the model in training mode
    model.train()

    # iterate over epochs
    for e in range(epochs):
        # iterate over training steps
        for i in tqdm(range(len(train_loader))):
          try:
              x = next(dataloader_iter)
          except StopIteration:
              dataloader_iter = iter(train_loader)
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
              val_iter = iter(val_loader)
              for j in range(val_steps):
                try:
                  x_test = next(val_iter)
                except StopIteration:
                  val_iter = iter(val_loader)
                  x_test = next(val_iter)

                x_test = {k: v.to(device) for k, v in x_test.items()}
                with torch.no_grad():
                    test_outputs = model(**x_test)
                val_loss += test_outputs.loss.item()
              validation_losses.append(val_loss / val_steps)
              print("Test loss: ", val_loss / val_steps)

    return losses, validation_losses


def plot_losses(train_losses, val_losses, path="outputs/loss_plot.png"):
    plt.plot(train_losses, label="Training Loss")
    plt.plot([i * 10 for i in range(len(val_losses))], val_losses, label="Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.show()

def construct_test_samples(dataset):
    samples = []
    for i in range(10):
        example = dataset["test"][i]
        cipher_type = example["cipher_type"]
        ciphertext = example[cipher_type]
        input_text = f"Identify the cipher used in the ciphertext:\n{ciphertext}"
        samples.append({"text": input_text, "label": cipher_type})
    return samples

def evaluate(model, tokenizer, samples):
    predictions = []
    model.eval()

    for sample in samples:
        inputs = sample["text"]
        inputs = tokenizer(inputs, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_class_id = logits.argmax(dim=-1).item()
        predicted_label = f"LABEL_{predicted_class_id}"

        predictions.append((sample["text"], predicted_label, sample["label"]))

    return predictions


def main():
    print("Device:", device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = prepare_dataset()
    tokenized_dataset = process_dataset(dataset, tokenizer)

    model = load_model()

    collate_fn = default_data_collator

    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["test"]

    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    train_steps = len(dataloader)

    train_losses, val_losses = train(
    model,
    dataloader,
    validation_dataloader,
    len(dataloader),
    5,
    150
)

    plot_losses(train_losses, val_losses)

    os.makedirs("saved_models", exist_ok=True)
    model.save_pretrained("saved_models/bert-cipher-classifier")
    tokenizer.save_pretrained("saved_models/bert-cipher-classifier")

    test_samples = construct_test_samples(dataset)
    predictions = evaluate(model, tokenizer, test_samples)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/predictions.txt", "w") as f:
        for i, (text, pred, true) in enumerate(predictions, 1):
            f.write(f"Example {i}:\nInput: {text}\nPredicted: {pred}\nTrue: {true}\n\n")

if __name__ == "__main__":
    main()
