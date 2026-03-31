from load_arc import load_arc_folder
import torch

train = load_arc_folder("data/training")
eval_ = load_arc_folder("data/evaluation")

torch.save(train, "data/arc_train.pt")
torch.save(eval_, "data/arc_eval.pt")
# print(train[0])
print("Saved processed dataset!")