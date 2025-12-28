### Plan to refactor the moe_flash_attn coding
1. Strategy: Unified Loss Module with Clean Separation of Concerns
The goal is:
- Single source of truth for loss computation
- Configurable (weighted/unweighted, with/without aux loss)
- Usable standalone or inside DataParallel

2. 

3. Isolate the optimizer, scheduler, training related configs from basic configs; 