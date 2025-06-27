This repo contains the source code of the paper accepted by ACL'2025 main - [**"Unveiling Privacy Risks in LLM Agent Memory"**](https://arxiv.org/pdf/2502.13172).

🚩 If possible, could you please star this project. ⭐ ↗️

## 1. Thanks
The repository is partially based on [EHRAgent](https://github.com/wshi83/EhrAgent) and RAP.

## 2. Attacking on EHRAgent
You can run the following files in sequence to conduct MEXTRA on EHRAgent. 
- _EHRAgent/attacking/init_memory.py_
  
  Initialize your memory. (Optional, you can directly use our provided memory.)

- _EHRAgent/attacking/attacking_prompt_generation.py_
  
  Automatically generate attacking prompts. (Optional, you can directly use our generated memory.)
- _EHRAgent/attacking/run_attack.py_
  
  Running memory extraction attack.
- _EHRAgent/attacking/run_attack.py_
  
  Evaluation.

## 3. Attacking on RAP
TBD
