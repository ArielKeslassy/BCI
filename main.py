import torch
import wandb
from data import EEGDataset
from conformer import Conformer

wandb.login()
wandb.init(project="BCI")

# load the MindBigData dataset
file_path = r"./MU_processed.txt"
dataset = EEGDataset(file_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# load the model
model = Conformer()

loss_fn = torch.nn.CrossEntropyLoss()

# training loop
for epoch in range(10):
    for x, y in loader:
        _, logits = model(x.unsqueeze(1))
        y_hat = torch.nn.functional.softmax(logits, dim=1)
        loss = loss_fn(y_hat, y)
        loss.backward()
        wandb.log({"loss'": loss.item()})
    model.save(f"model_epoch_{epoch}.pt")






