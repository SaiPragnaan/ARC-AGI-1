import torch
import torch.nn.functional as F
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join('../..')))

from src.models.arc_model import ARCModel
from src.utils.config import load_config

import random
seed_value = 42
random.seed(seed_value)

def split_train_val(dataset:list):
    random.shuffle(dataset)
    
    all_task_id_list=[]
    for sample in dataset:
        all_task_id_list.append(sample["task_id"])
    filtered_task_id_list=[x for x in all_task_id_list if all_task_id_list.count(x)==1]
    
    eval_dataset=[]
    eval_task_ids=[]
    
    for i, sample in enumerate(dataset):
        if sample['task_id'] not in filtered_task_id_list:
            continue
        if len(eval_dataset) >= 35:
            break
        if sample['task_id'] in eval_task_ids:
            continue
        eval_dataset.append(sample)
        eval_task_ids.append(sample['task_id'])

    train_dataset = [s for s in dataset if s['task_id'] not in eval_task_ids]

    return train_dataset, eval_dataset


def exact_match(pred, target):
    return (pred == target).all().item()


if __name__=="__main__":

    os.makedirs("../../checkpoints", exist_ok=True)
    
    train_dataset=torch.load('../../data/arc_train.pt')
    test_dataset=torch.load('../../data/arc_eval.pt')
    
    train_dataset,eval_dataset=split_train_val(train_dataset)
    # print("PRE LEN of TRAIN DATASET : ",pre_len)
    # print("LEN of TRAIN DATASET : ",len(train_dataset))
    # print("LEN of EVAL DATASET  : ",len(eval_dataset))
    
    config=load_config('../../configs/baseline.yaml')
    wandb.init(
        project="ARC-AGI-1",
        config=config
    )
    
    device="cuda" if torch.cuda.is_available() else "cpu"

    model=ARCModel(config=config["model"]).to(device)
    
    lr=config["training"]["lr"]
    batch_size=config["training"]["batch_size"]
    num_epochs=config["training"]["epochs"]
    weight_decay=config["training"]["weight_decay"]
    
    optimizer=torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
    
    total_steps = num_epochs * len(train_dataset)
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        return max(0.0, (total_steps - current_step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_eval_acc=-1
    
    for epoch in range(num_epochs):
        random.shuffle(train_dataset)

        model.train()
        per_epoch_train_loss=0
        per_epoch_train_acc=0
        
        for sample in train_dataset:
            demos = sample["demos"]
            demos = [(x.to(device), y.to(device)) for x, y in demos]
            test_input = sample["test_input"].to(device)
            target = sample["target"].to(device)

            outputs=model(demos, test_input)
            loss = sum(F.cross_entropy(out.view(-1, 10), target.view(-1)) for out in outputs)  # TODO : Experiment (i+1)* type shi here, as later steps of refinement matter more than starting

            optimizer.zero_grad()
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            
            pred = outputs[-1].argmax(-1)
            per_epoch_train_acc += exact_match(pred.view_as(target), target)
            per_epoch_train_loss+=loss.item()
            
            wandb.log({
                "per_iter_train_loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm
                })
            
        per_epoch_train_loss/=len(train_dataset)
        per_epoch_train_acc/=len(train_dataset)
        
        model.eval()
        with torch.inference_mode():
            per_epoch_eval_loss=0
            per_epoch_eval_acc=0
            
            for sample in eval_dataset:
                demos = sample["demos"]
                demos = [(x.to(device), y.to(device)) for x, y in demos]
                test_input = sample["test_input"].to(device)
                target = sample["target"].to(device)
            
                outputs=model(demos, test_input)
                loss = sum(F.cross_entropy(out.view(-1, 10), target.view(-1)) for out in outputs) / len(outputs)
                
                pred = outputs[-1].argmax(-1)
                per_epoch_eval_acc += exact_match(pred.view_as(target), target)
                per_epoch_eval_loss+=loss.item()

            per_epoch_eval_loss/=len(eval_dataset)
            per_epoch_eval_acc/=len(eval_dataset)

        print(f"Epoch {epoch} | Train Loss: {per_epoch_train_loss:.4f} | Eval Loss: {per_epoch_eval_loss:.4f} | Train Acc: {per_epoch_train_acc:.4f} | Val Acc: {per_epoch_eval_acc:.4f}") 
        wandb.log({
            "epoch": epoch,
            "per_epoch_train_loss": per_epoch_train_loss,
            "per_epoch_eval_loss": per_epoch_eval_loss,
            "train_acc": per_epoch_train_acc,
            "val_acc": per_epoch_eval_acc
            })

        if per_epoch_eval_acc >= best_eval_acc :
            best_val_acc = per_epoch_eval_acc
            torch.save(model.state_dict(), "../../checkpoints/best_model.pt")

    torch.save(model.state_dict(), "../../checkpoints/last_model.pt")