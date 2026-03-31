import os
import torch
import json

def load_arc_folder(folder_path):
    samples=[]
    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue
        path=os.path.join(folder_path,file)
        with open(path,'r') as f:
            data=json.load(f)

        demos=[]
        for pair in data["train"]:
            x = torch.tensor(pair["input"], dtype=torch.long)
            y = torch.tensor(pair["output"], dtype=torch.long)
            demos.append((x, y))

        for i, test_pair in enumerate(data["test"]):

            sample = {}
            sample["task_id"] = file.replace(".json", "")
            sample["test_id"] = i
            sample["demos"] = demos

            sample["test_input"] = torch.tensor(test_pair["input"], dtype=torch.long)
            sample["num_demos"] = len(demos)
            
            if "output" in test_pair:
                sample["target"] = torch.tensor(test_pair["output"], dtype=torch.long)
            else:
                sample["target"] = None

            samples.append(sample)

    return samples
