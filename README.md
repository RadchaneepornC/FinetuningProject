# Typhoon Instruction tuning with WikiThai-V03 dataset

## Motivation
As fine-tuning techniques become popular for enhancing various downstream applications, Thai Language Models (LLMs) have been developed for low-resource languages. Simultaneously, high-quality datasets have been released. I have the idea to experiment with improving LLMs for Question-Answering tasks by fine-tuning them with the dataset

## Dataset for finetuning


## Fine-tuning process

## Finetuning Results 

## Analysis & further study for improving 

- **Clean Dataset**
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


- **Try experiment finetuning using other Pretained LLMs understanding Thai, which are listed below:**
  


## Resource

**For Finetuning tutorial**

- https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca
- https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb

**For inferencing model**
- https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge?utm_source=%2Fsearch%2Fqlora&utm_medium=reader2

**For original dataset**
- https://huggingface.co/datasets/pythainlp/thai-wiki-dataset-v3
