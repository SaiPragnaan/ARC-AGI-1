import torch
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join('../..')))

from src.models.arc_model import ARCModel
from src.utils.config import load_config

def train(model, dataloader, optimizer, config):
    model.train()

    for batch in dataloader:
        demos = batch["demos"]
        test_input = batch["test_input"]
        target = batch["target"]

        outputs = model(demos, test_input)

        loss = 0
        for out in outputs:
            loss += torch.nn.functional.cross_entropy(           # TODO : Experiment (i+1)* type shi here, as later steps of refinement matter more than starting
                out.view(-1, 10),
                target.view(-1)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})

if __name__=="__main__":

    train_dataset=torch.load('../../data/arc_train.pt')
    test_dataset=torch.load('../../data/arc_eval.pt')
    
    config=load_config('../../configs/baseline.yaml')
    device="cuda" if torch.cuda.is_available() else "cpu"

    model=ARCModel(config=config["model"]).to(device)
    
    lr=config["training"]["lr"]
    batch_size=config["training"]["batch_size"]
    num_epochs=config["training"]["epochs"]
    
    optimizer=torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=1e-4
        )
    # for epoch in num_epochs:
    #     model.train()
    #     total_epoch_loss=0
    #     for sample in train_dataset:
    #         demos = sample["demos"]
    #         test_input = sample["test_input"]
    #         target = sample["target"]
        
    #         outputs=model(demos, test_input)
            