## Experiment 1 (baseline or baseline++)
**Config Used          :** `configs/baseline.yaml`

**Number of Parameters :** ~8.5M
### Architecture : 
```
Demos (x₁,y₁), (x₂,y₂), ... 
        ↓
Grid Embedding (color + row + col + pair_id + io_flag)
        ↓
Encoder (Transformer, 6 layers)
        ↓
Latent Tokens (K = 16) via cross-attention → z₀
        ↓
Test Input → Embedding → x
```
Basic Overview of refinement block
```python
Initialize y₀ = zeros
TRM Loop:
    for N_sup steps:
        for n_z steps:
            z ← TRMBlock(z, [x, y])
        y ← TRMBlock(y, [x, z])

        logits ← Linear(y)
Final prediction = last refinement step
```

### Design Choices :
- ***Grid Representation***
    - Each cell encoded using learnable embeddings (color + positional + structural signals)

    - Added **pair_id** and **IO flag** to distinguish demos vs test and input vs output
- ***Latent Rule Representation (z)***
    - Fixed number of latent tokens (K = 16)

    - Extracted via cross-attention over encoded demo features
    - Acts as compressed “rule representation”
- ***TRM-style Iterative Refinement***
    - Shared TRM block used for both z and y updates (parameter-efficient)

    - Separation of:
        - z updates → reasoning
        - y update → prediction
- ***Depth inside TRM Block***
    - Each update consists of a stack of transformer layers (depth = 4)
- ***Training Strategy***
    - Deep supervision: loss computed at every refinement step

    - Equal weighting across steps (baseline choice)
- ***Optimization***
    - AdamW optimizer with warmup + linear decay scheduler

    - Gradient clipping applied for stability
- ***Evaluation Metric***
    - Exact match accuracy (strict ARC metric)
    
### Limitations :
- Does not deal with Batching yet.. aka works for batch_size=1, which is not the best, which should be solved in further experiments.

- Using x + z in y-update deviates from strict TRM formulation
- No data augmentation or task-level curriculum
- No test-time adaptation (TTT), which is known to help ARC


### Next TODOs in upcoming experiments :
- Fiddle with Embedding dimension 

- Fiddle with K
- Increase `n_z`, `N_sup`
- Increasing the model params
- Currently the loss is added directly without any weights, we may try multiplying a weight which would be more for later refinement steps, and less for earlier
- In the y-update step after `n_z` z-updates, we are taking both x,z to update y, but in the original TRM paper they are using only z.
- Implement **proper batching with padding + masking**
