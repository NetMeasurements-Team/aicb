## Pipeline Parallelism

The workload generator assumes that the number of GPUs provided as input is the total one, including multiple pipeline 
 groups. Therefore `DP = world_size / (TP * PP)`

```python
args.dp_num = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel)
```

Same assumption accounts for number of layers:

```python
if args.pipeline_model_parallel > 1 :
    args.num_layers = int(args.num_layers//args.pipeline_model_parallel)
```

However, SimAI does not consider this and assumes that the world size in the workload is `DP * TP`.
There are two ways to do this correctly, in both cases SimAI should compute `DP = world_size / (PP * TP)`:
1. **LARGE SIMULATION:** simulate all the `PP` stages (not very useful since pp communication is not simulated)
2. **SMALL SIMULATION:** only simulate one pipeline stage (`world_size / PP`) and ignore `PP`. In this case the topology passed 
   to SimAI should only feature `world_size / PP` GPUs.

### Output workload file

> **PROBLEM:** only one output workload file is generated. If the layers are not all the same throughout the model 
  (e.g., DeepSeek), the workload will only be representative of the first stage of the pipeline. Additionally, it will 
  always comprehend initial and final extra layers, which might not be representative of all the pipeline stages.

In the output workload file:
- `all_gpus:` is the `world_size`, featuring all the `PP` stages
- `pp:` is the pipeline parallel size
- `vpp:` is the number of layers for a single pipeline group (which can be double-checked with the lines between two 
  `embedding_layer` in the file)
- `ga:` is the number of iterations (gradient accumulation) the forward and then backward should be repeated to reach
  the desired global_batch size (note that this does not change with `PP` if the `world_size` is adjusted accordingly)
- `pp_comm:` is the size in bytes of the intermediate tensors that should be exchanged between different pipeline stages
