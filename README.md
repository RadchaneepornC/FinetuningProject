# Typhoon Instruction tuning with WikiThai-V03 dataset using QLORA technique

<p align="center">
  <img src="https://github.com/RadchaneepornC/FinetuningProject/blob/main/Images/picture.jpg" alt="Alt text">
</p>

[credit of picture](https://i.pinimg.com/originals/12/31/a4/1231a468460cea163fd26f2f6eda86bc.jpg)

## Motivation
As fine-tuning techniques become popular for enhancing various downstream applications, Thai Language Models (LLMs) have been developed for low-resource languages. Simultaneously, high-quality datasets have been released. I have the idea to experiment with improving LLMs for Question-Answering tasks by fine-tuning them with the specific domain dataset

## Dataset for finetuning
For this experiment, I pass each prompt to mistral prompt template


      <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
      NOTE <s> </s> are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings.

I select only ```title``` and ```text``` from the original dataset, then I add new ```instruction``` column including the sentence "อธิบายความหมายและให้ข้อมูลของคำดังต่อไปนี้" according to source of dataset is from the wikipedia which is the website that people usually search for definition of words

below are finalised dataframe before passing to mistral chat prompt template
| Instruction | Input | Output| 
|-------------------|------|--------------|
|อธิบายความหมายและให้ข้อมูลของคำดังต่อไปนี้|วันจันทร์| วันจันทร์ เป็นวันลำดับที่ 2 ในสัปดาห์ อยู่ระหว่าง...|

Also, this is the [source code](https://github.com/RadchaneepornC/FinetuningProject/blob/main/Dataset_preprocessing.ipynb) for passing data to mistral chat prompt template

## Fine-tuning process

See overall steps in this [document](https://github.com/RadchaneepornC/FinetuningProject/blob/main/Finetuning%20code.ipynb)

**0. Check GPU**

```python

! nvidia-smi

```
**NOTE** For this time, I used V100 GPU running apploximately 4 hours, below is my usaged resources for this overall finetuning process
![Alt text](https://github.com/RadchaneepornC/FinetuningProject/blob/main/Images/Resource_usage.png)

**1. Install require packages**
```python
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q trl
!pip install -q  wandb datasets
!pip install huggingface-hub==0.19.4
!pip install gradio
```

**2. Preprocessing data for finetuning**

```python

from datasets import load_dataset
train_dataset = load_dataset('json', data_files='/content/drive/MyDrive/ThaiDataset/wikiThai_ft_Typhoon.jsonl', split='train')
```

**3. Load the model to GPU memory in 4-bit**

```python

#import requirement libary
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
import torch

# Load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

base_model = "scb10x/typhoon-7b"
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
model.config.use_cache = False 
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.bos_token, tokenizer.eos_token
```

**4. Define the LoRA configuration**

```python
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

#Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)

#Setting LORA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)
```

**5. Define training arguments**

```python
#import requirement libary
from transformers import TrainingArguments

#Hyperparamter setting
training_arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/FinetuneTyphoon",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

```
**6. Pass the arguments defined in step 4 into an instance of SFTTrainer**
```python
# Import require libaries
from trl import SFTTrainer
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="Text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

#finetuning model
trainer.train()
```

**7. Save the adapter we got from trainer**

```python
trainer.model.save_pretrained(new_model)
model.config.use_cache = True
model.eval()

#logging in the huggingface
from google.colab import userdata
secret_hf = userdata.get('hf_for_write_model')
!huggingface-cli login --token $secret_hf

#push Adaptor to huggingface
new_model = "Typhoon_wikiThai"
trainer.model.push_to_hub(new_model)
```


**8. Test finetuned model**
```python
from transformers import pipeline, logging
logging.set_verbosity(logging.CRITICAL)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

def build_prompt(question):
  prompt=f"<s>[INST]{question}[/INST]"
  return prompt

question = input()
prompt = build_prompt(question)
result = pipe(prompt)

print(result[0]['generated_text'])
```

**9. Inference the based model with finedtuned adaptor(for later use)**

```python
base_model = "scb10x/typhoon-7b"
new_model = "radchaneepornc/Typhoon_wikiThai"

#load model
base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
)

#model + LoRA adaptor
model = PeftModel.from_pretrained(base_model_reload, new_model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

#create pipeline for using model
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

#create UI for easier using
import gradio as gr
def chatbot(user_input):
    if user_input.lower() == 'exit':
        return "Exiting."
    return f"{pipe(user_input)[0]['generated_text']}"
iface = gr.Interface(fn=chatbot, inputs="text", outputs= "text")
iface.launch()
```


## Finetuning Results 
For this project, I randomly generate and evaluate responses using human evaluation to assess the relevance of each answer to each prompt
| Prompt | Answer from Typhoon | Answer from finetuned Typhoon | 
|-------------------|------|--------------|
| ดาราศาสตร์ คืออะไร| ดาราศาสตร์ คือ การศึกษาเกี่ยวกับวัตถุในอวกาศ เช่น ดาวเคราะห์ ดวงดาว ดาวฤกษ์ ดาวหาง ดาวเคราะห์น้อย และอื่นๆ\nดาราศาสตร์ เป็นวิชาที่ศึกษาเกี่ยวกับวัตถุในอวกาศ เช่น ดาวเคราะห์ ดวงดาว ดาวฤกษ์ ดาวหาง ดาวเคราะห์น้อย และอื่นๆ\nดาราศาสตร์ เป็นวิชาที่ศึกษาเกี่ยวกับวัตถุในอวกาศ เช่น ดาวเคราะห์ ดวงดาว ดาวฤกษ์ ดาวหาง ดาวเคราะห์น้อย และอื่นๆ\nดาราศาสตร์ เป็นวิชาที่ศึกษาเกี่ยวกับวัตถุในอวกาศ เช่น ดาวเคราะห์ ดวงดาว ดาวฤกษ์ ดาวหาง ดาวเคราะห์น้อย และอื่นๆ\nดาราศาสตร์ เป็นวิชาที่ศึกษาเกี่ยวกับวัตถุในอวกาศ เช่น ดาวเคราะห์ ดวงดาว ดาวฤกษ์ ดาวหาง ดาวเคราะห์น้อย และอื่นๆ\nดาราศาสตร์ เป็นวิชาที่ศึกษาเกี่ยวกับวัตถุในอวกาศ เช่น   | ดาราศาสตร์ (astronomy) คือการศึกษาเกี่ยวกับวัตถุท้องฟ้า เช่น ดาวฤกษ์ ดาวเคราะห์ ดาวหาง ดาวตก ดาวตกสะเก็ด ดาวตกหาง ดาวเคราะห์น้อย ดาวเคราะห์แคระ ดาวเคราะห์น้อย ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์แคระ ดาวเคราะห์ |
|วิศวะคอมพิวเตอร์ คืออะไร|วิศวกรรมคอมพิวเตอร์ (อังกฤษ: Computer Engineering) เป็นสาขาวิชาที่ศึกษาเกี่ยวกับการประยุกต์ใช้ความรู้ทางด้านวิศวกรรมศาสตร์และคณิตศาสตร์ เพื่อสร้างอุปกรณ์อิเล็กทรอนิกส์และซอฟต์แวร์ที่ทำงานร่วมกัน วิศวกรรมคอมพิวเตอร์เป็นสาขาวิชาที่ผสมผสานระหว่างวิศวกรรมไฟฟ้าและวิทยาการคอมพิวเตอร์เข้าด้วยกัน วิศวกรรมคอมพิวเตอร์เป็นสาขาวิชาที่เน้นการประยุกต์ใช้ความรู้ทางด้านวิศวกรรมศาสตร์และคณิตศาสตร์ เพื่อสร้างอุปกรณ์อิเล็กทรอนิกส์และซอฟต์แวร์ที่ทำงานร่วมกัน วิศวกรรมคอมพิวเตอร์เป็นสาขาวิชาที่ผสมผสานระหว่างวิศวกรรมไฟฟ้าและวิทยาการคอมพิวเตอร์เข้าด้วยกัน วิศวกรรมคอมพิวเตอร์เป็นสาขาวิชาที่เน้นการประยุกต์ใช้ความรู้ทางด้านวิศวกรรมศาสตร์และคณิตศาสตร์ เพื่อสร้างอุปกรณ์อิเล็กทรอนิกส์และซอฟต์แวร์ที่ทำงานร่วมกัน|วิศวกรรมคอมพิวเตอร์ (computer engineering) เป็นสาขาวิชาที่ศึกษาเกี่ยวกับการออกแบบและพัฒนาอุปกรณ์อิเล็กทรอนิกส์และซอฟต์แวร์ที่เกี่ยวข้องกับคอมพิวเตอร์ วิศวกรรมคอมพิวเตอร์เป็นสาขาวิชาที่ผสมระหว่างวิศวกรรมไฟฟ้าและวิศวกรรมซอฟต์แวร์\n== ดูเพิ่ม ==\nวิศวกรรมคอมพิวเตอร์\nวิศวกรรมไฟฟ้า\nวิศวกรรมซอฟต์แวร์ \n== แหล่งข้อมูลอื่น ==\nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \nวิศวกรรมคอมพิวเตอร์ \n|
|การประมวลผลสัญญาณดิจิทัล คืออะไร|การประมวลผลสัญญาณดิจิทัล คือ การแปลงสัญญาณดิจิทัลให้เป็นสัญญาณอนาล็อก และสัญญาณอนาล็อกให้เป็นสัญญาณดิจิทัล\nการประมวลผลสัญญาณดิจิทัล คือ การแปลงสัญญาณดิจิทัลให้เป็นสัญญาณอนาล็อก และสัญญาณอนาล็อกให้เป็นสัญญาณดิจิทัล\nร่วมเป็นแฟนเพจเรา บน Facebook..ได้ที่นี่เลย!! ▼ กำลังโหลดข้อมูล... ▼\nแสดงความคิดเห็น\nกระทู้ที่คุณอาจสนใจ\nอยากทราบว่ามีใครเคยใช้บริการของบริษัทนี้ไหมคะ\nอยากทราบว่ามีใครเคยใช้บริการของบริษัทนี้ไหมคะ พอดีเราได้ไปสมัครงานกับบริษัทนี้มาค่ะ แล้วเขาให้เราไปทำใบรับรองแพทย์กับโรงพยาบาลที่เขาแนะนำมาให้ค่ะ พอเราไปทำใบรับรองแพทย์เสร็จแล้ว|การประมวลผลสัญญาณดิจิทัล (digital signal processing) หรือเรียกย่อ ๆ ว่า การประมวลผลสัญญาณ (signal processing) คือ การประมวลผลสัญญาณดิจิทัล (digital signal) โดยใช้เทคนิคต่าง ๆ เช่น การวิเคราะห์สัญญาณ (signal analysis) การสังเคราะห์สัญญาณ (signal synthesis) การประมวลผลสัญญาณเสียง (audio signal processing) การประมวลผลสัญญาณภาพ (image signal processing) การประมวลผลสัญญาณวิดีโอ (video signal processing) การประมวลผลสัญญาณแบบสุ่ม (stochastic signal processing) การประมวลผลสัญญาณแบบไม่สุ่ม (deterministic signal processing) การประมวลผลสัญญาณแบบอนุกรม (sequential signal processing) การประมวลผลสัญญาณแบบขนาน (parallel signal processing) การประมวลผลสัญญาณแบบเวลาจริง (real-time signal processing) การประมวลผลสัญญาณแบบ|
|วันศุกร์คืออะไร|วันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุกร์คืออะไร\nวันศุก|วันศุกร์ (Friday) เป็นวันหนึ่งในสัปดาห์ เป็นวันที่ 5 ของสัปดาห์ ตามหลักโหราศาสตร์ เป็นวันสุดท้ายของสัปดาห์ และเป็นวันสุดท้ายของวันทำงานของชาวตะวันตก\n== วันศุกร์ในภาษาต่าง ๆ ==\n== ดูเพิ่ม ==\nรายชื่อวันในสัปดาห์\n== อ้างอิง ==\n== แหล่งข้อมูลอื่น ==\nวันศุกร์\nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์ \nวันศุกร์|

## Analysis & further study for improving 

- **Clean Dataset**

I formatted the data but didn't clean it properly. I need to address the syntax issues

- **Finetuned with more examples**

For this case, I finetuned with 3000 examples, Increasing size of examples for finetuning is prospect plan for my next experiment 

- **Chat Prompt Template**

I am uncertain whether experimenting with changing the prompt template from <br>```<s>[INST] {instruction} here are the inputs {input_text} [/INST] \\n {output} </s>``` <br> to a completely Thai version like <br> ```<s>[INST] {instruction} ความหมายของ {input_text} คือ [/INST] \\n {output} </s>``` <br>will have any effect on the model's performance


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
|Sailor (0.5B, 1.8B, 4B and 7B)| Sea AI Lab |2 Mar 2024| Qwen1.5| outperforms SeaLLM-7B V1 & V2 and thier based models in many evaluation tasks|
|OpenThaiGPT-13B (version 1.0.0-beta)| AIEAT, AIAT, NECTEC, et al| 20 Dec 2023|LLaMA v2 Chat (13b)| |

## Conclusions

As I try to experiment with improving the response quality from Thai open-source LLMs by fine-tuning them with domain-specific datasets, and evaluate the relevance of the responses by randomly prompting the model and comparing the answers with those from the off-the-shelf Typhoon in the same prompt, we found that we could not conclusively determine whether the model's answers have significantly improved. While some prompts yielded better answers than the off-the-shelf model, there is still ample room for improvement in the model. Further steps to enhance the model include cleaning the dataset more thoroughly, adjusting the chat template prompts, increasing the number of examples for fine-tuning, adjusting hyperparameter configurations, and changing the method of loading the base model during the inference step



## Resources

**For Finetuning tutorial**

- https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca
- https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
- https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms

**For inferencing model**
- https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge?utm_source=%2Fsearch%2Fqlora&utm_medium=reader2

**For based model**
- https://huggingface.co/scb10x/typhoon-7b
  
**For original dataset**
- https://huggingface.co/datasets/pythainlp/thai-wiki-dataset-v3
