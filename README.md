# Typhoon Instruction tuning with WikiThai-V03 dataset

## Motivation
As fine-tuning techniques become popular for enhancing various downstream applications, Thai Language Models (LLMs) have been developed for low-resource languages. Simultaneously, high-quality datasets have been released. I have the idea to experiment with improving LLMs for Question-Answering tasks by fine-tuning them with the dataset

## Dataset for finetuning



## Fine-tuning process

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

Since I formated the data,but did not further clean them properly, I need to clean the syntax  

- **Chat Prompt Template**
For this experiment, I pass each prompt to mistral prompt template


      <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
      NOTE <s> </s> are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings.

I select only ```title``` and ```text``` from the original dataset, then I add new ```instruction``` column including the sentence "อธิบายความหมายและให้ข้อมูลของคำดังต่อไปนี้" according to source of dataset is from the wikipedia which is the website that people usually search for definition of words

below are finalised dataframe before passing to mistral chat prompt template



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




## Resource

**For Finetuning tutorial**

- https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca
- https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb

**For inferencing model**
- https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge?utm_source=%2Fsearch%2Fqlora&utm_medium=reader2

**For original dataset**
- https://huggingface.co/datasets/pythainlp/thai-wiki-dataset-v3
