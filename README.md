# reasoning
Fine-tuning an LLM to be a reasoning model using GRPO reinforcement learning.


![Untitled](https://github.com/user-attachments/assets/a0fbdfe4-f796-4a82-b0a8-035318b6e8d2)  
This code uses the Group Relative Policy Optimization (GRPO) reinforcement learning (RL) method invented by the Deepseek team, as described in:  

DeepSeekMath:  
https://arxiv.org/abs/2402.03300  

DeepSeek-R1:  
https://arxiv.org/abs/2501.12948  


# Some resources:
Why GRPO is Important and How it Works:  

https://www.oxen.ai/blog/why-grpo-is-important-and-how-it-works  
https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo  

These Videos are an amazing resource:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/90ImcYM0xWc/0.jpg)](https://www.youtube.com/watch?v=90ImcYM0xWc)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-7Y4s7ItQQ4/0.jpg)](https://www.youtube.com/watch?v=-7Y4s7ItQQ4)

# The QLoRA Fine Tuning 

The fine-tuning was made much easier by using unsloth.ai:

[![Alt text](https://github.com/user-attachments/assets/303f049a-ea6b-48ac-ab08-ab85f3ed2384)](https://unsloth.ai)

Specifically, unsloth leverages the GRPOTrainer class from TRL:

https://huggingface.co/docs/trl/main/en/grpo_trainer



