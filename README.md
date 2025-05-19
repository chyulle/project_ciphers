## Project for Fine-tuning Bert for Ciphers Recognition

The learning goal is to practice fine-tuning a pretrained LM, in this scenario Bert, for a text recognition task, using task-specific dataset, [CipherBank]([https://huggingface.co/datasets/tau/commonsense_qa](https://huggingface.co/datasets/yu0226/CipherBank/viewer/Atbash?views%5B%5D=atbash)). The performance of the model in evaluated on test split of the dataset over training to monitor whether the model's performance is improving and compare the performance of the base pretrained model and the fine-tuned model.
The following steps are performed:

1. Preparing data and a custom Dataset
2. Loading the pretrained model
3. Setting up training pipeline
4. Running the training, while tracking the losses
5. Output analysis
