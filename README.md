# Typhoon Instruction tuning with WikiThai-V03 dataset

## Motivation
As fine-tuning techniques become popular for enhancing various downstream applications, Thai Language Models (LLMs) have been developed for low-resource languages. Simultaneously, high-quality datasets have been released. I have the idea to experiment with improving LLMs for Question-Answering tasks by fine-tuning them with the dataset

## Dataset for finetuning


## Fine-tuning process

## Finetuning Results 

## Analysis & further study for improving 

- **Clean Dataset**

Since I formated the data,but did not further clean them properly, I need to clean the syntax  
- **Chat Prompt Pemplate**
  
- **Overall finetuning technique**
    - The loss curve is monitored from the training set using Weight and Bias (WandB)
  ![Alt Text](https://github.com/RadchaneepornC/FinetuningProject/blob/main/Images/trainloss_wandb.png)
 The training graph suggests overfitting since the training loss fluctuates between 0.5 and 2, this may from factors like insufficiently preprocessing of the data, suboptimal hyperparameter configuration, etc.

- **Inference technique**
  - Since I have found from this experimental blog about [**LoRA Adapters: When a Naive Merge Leads to Poor Performance**](https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge?utm_source=%2Fsearch%2Fqlora&utm_medium=reader2), the article show result that

> We have to load the base model and preprocess it the same way it was during QLoRA fine-tuning, otherwise, we may get a significant performance drop. The same applies if want to merge the adapter.
Then I thought that the way I load base-model without BitsAndBytesConfig setting for inferencing could be one of root causes of poor performance of my model
<br>

Use the code below to load the model based on the next time to incorporate with an adapter fine-tuned in 4-bit precision

```python
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")

```

- **Try experiment finetuning using other Pretained LLMs understanding Thai, which are listed below:**


| Pre-trained Model | Team | Release time | Foundation model | Performance |
|-------------------|------|--------------|------------------|-------------|
| SeaLLM-7B-v2|DAMO Academy, Alibaba Group| 1 Feb 2024| Mistral-7B-v0.1|outperforms GPT-3.5 and Qwen-14B on the multilingual MGSM for Zh and Th in zero shot|
|Sailor (0.5B, 1.8B, 4B and 7B)| Sea AI Lab |2 Mar 2024| Qwen1.5| outperforms SeaLLM-7B V1 & V2 and thier based model in many evaluation tasks|
|OpenThaiGPT-13B (version 1.0.0-beta)| AIEAT, AIAT, NECTEC, et al| 20 Dec 2023|LLaMA v2 Chat (13b)| |




## Resource

**For Finetuning tutorial**

- https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca
- https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb

**For inferencing model**
- https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge?utm_source=%2Fsearch%2Fqlora&utm_medium=reader2

**For original dataset**
- https://huggingface.co/datasets/pythainlp/thai-wiki-dataset-v3
