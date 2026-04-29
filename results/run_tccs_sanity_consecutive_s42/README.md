---
library_name: peft
model_name: run_tccs_sanity_consecutive_s42
tags:
- base_model:adapter:/home/jqh/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c
- sft
- transformers
- trl
licence: license
base_model: /home/jqh/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c
---

# Model Card for run_tccs_sanity_consecutive_s42

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.1.dev0
- TRL: 0.23.0
- Transformers: 4.57.2
- Pytorch: 2.8.0+cu128
- Datasets: 4.7.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```