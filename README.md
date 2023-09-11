# Fine-tuning 4-bit Llama-2-7b with Flash Attention using DPO


This is essentially a documentation of the training process of 4-bit llama-2–7b model which I was trying to fine-tune on Stack-exchange dataset using DPO, **but for some reason, the training prematurely stopped in between and since I’m running low on compute hence I’m not able to retrain at the moment. However, I hope to get it trained as soon as possible.**

Now let’s begin!

## What is DPO?
DPO offers a streamlined method for optimizing human-derived preferences in LLMs, such as GPT-4 or Claude. Traditional models utilize reinforcement learning (RL) to train models based on human feedback. This process, known as Reinforcement Learning from Human Feedback (RLHF), involves building a good reward function and carefully training the model to produce sensible text that aligns with human expectations.

DPO simplifies the RLHF process. Instead of using RL and a reward model, DPO employs a direct binary cross-entropy loss to fine-tune models. This method is significantly more straightforward and eliminates many complexities associated with RLHF.

## How does DPO differ from PPO?
While the traditional RLHF method uses an auxiliary reward model to fine-tune models, DPO skips the reward modeling step entirely. It uses an analytical mapping from the reward function to the optimal RL policy, allowing for direct optimization of the language model based on preference data. Essentially, DPO simplifies the optimization process by focusing on the reference model and omitting the need for RL-based optimization.

## Training with TRL’s DPO:
The TRL library, which supports DPO, provides tools for the entire RLHF pipeline. However, with DPO, only supervised fine-tuning (SFT) and data annotation for preference labels are needed. The DPOTrainer in TRL optimizes the model using the preference data.

For example, to train with the Stack Exchange preference dataset, you’d need to format the data appropriately. After processing the data, you can use the DPOTrainer, which needs the base model from the SFT pipeline, a reference model, and other necessary parameters to begin training.

So in short, for training with TRL’s DPO we will need to do the following three steps:

1. a supervised fine-tuning (SFT) step
2. the process of annotating data with preference labels
3. provide the DPOTrainer in TRL with preference data from step 2 which has a very specific format

Precisely, the data should be in a dictionary format with the following three keys: context prompt, chosen response, and rejected response.

