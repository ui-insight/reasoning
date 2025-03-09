# Training a Mathematical Reasoning Model with GRPO
QLoRA fine-tuning of an LLM to be a reasoning model using GRPO reinforcement learning.  
Trained in < 24 hours on an H100 GPU against the [GSM8K](https://github.com/openai/grade-school-math) dataset.  

  

![Untitled](https://github.com/user-attachments/assets/a0fbdfe4-f796-4a82-b0a8-035318b6e8d2)  
This code uses the Group Relative Policy Optimization (GRPO) reinforcement learning (RL) method invented by the Deepseek team, as described in:  

DeepSeekMath:  
https://arxiv.org/abs/2402.03300  

DeepSeek-R1:  
https://arxiv.org/abs/2501.12948  

<img width="1197" alt="image" src="https://github.com/user-attachments/assets/72be5d9a-71e5-460f-878d-103562b80585" />


  
# Some resources:
Why GRPO is Important and How it Works:  

https://www.oxen.ai/blog/why-grpo-is-important-and-how-it-works  
https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo  

These Videos are an amazing resource:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/90ImcYM0xWc/0.jpg)](https://www.youtube.com/watch?v=90ImcYM0xWc)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-7Y4s7ItQQ4/0.jpg)](https://www.youtube.com/watch?v=-7Y4s7ItQQ4)

  

## The Math Behind GRPO
https://medium.com/yugen-ai-technology-blog/understanding-the-math-behind-grpo-deepseek-r1-zero-9fb15e103a0a


# The QLoRA Fine Tuning 

The fine-tuning was made much easier by using https://unsloth.ai:

[![Alt text](https://github.com/user-attachments/assets/303f049a-ea6b-48ac-ab08-ab85f3ed2384)](https://unsloth.ai)

Specifically, unsloth leverages the GRPOTrainer class:

https://huggingface.co/docs/trl/main/en/grpo_trainer  
  
from the Transformer Reinforcement Learning (TRL) package:   
  
https://huggingface.co/docs/trl/en/index

  
Finally, the folks at [unsloth](https://unsloth.ai) have a great [blog post](https://unsloth.ai/blog/r1-reasoning) and [Google Collab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) where they do something very similar.



