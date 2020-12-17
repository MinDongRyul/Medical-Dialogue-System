### How to Run

1. As for the dataset, you can get access to the raw dialogue dataset by download from [here](https://drive.google.com/file/d/1hRZdfvE8I6bevPjtvZVM6R109cOYRjVq/view?usp=sharing).
  In addition, we clean and split the raw dialogues, and the result can be download from [train_data.json](https://drive.google.com/file/d/1j2r6tLn3hRpt6ue7RZQv4Y4s9rB-VQ0J/view?usp=sharing), [validate_data.json](https://drive.google.com/file/d/1mxYRm8jwU3J2ztE495QXV4R3WFy3ESsm/view?usp=sharing) and [test_data.json](https://drive.google.com/file/d/1yAlR35gIrloXU6Rx4wuEuKQmXDTpczpB/view?usp=sharing).

  > Before running the code, you need to run preprocess.py with the json file or download easily to get the preprocessed dataset by [here](https://drive.google.com/file/d/1ZeUjmymeRMvVzGuyYz8kPFyzSwQs7BuB/view?usp=sharing) and move preprocessed files into this directory. 


(As for the original pretrained model, you need download it from [here](https://drive.google.com/file/d/17ywZ4LJNukGJNiMuGcIeSqtcBRiiJUnB/view?usp=sharing).
2. When training, run `python bert_gpt_train.py`. (We also provide our trained model with the meddialogue at [here](https://drive.google.com/file/d/1alyU4wEClpjj2-kGl45xxUal0dHZGZhI/view?usp=sharing).


3. When calculate the perplexity, run

   ```shell
   $ python bert_gpt_perplexity.py --decoder_path ${the path of the model you have saved}
   ```


4. When testing, run `python generate.py --decoder_path ${the path of the model you have saved}` to get the generated dialogues file,
and run `python validate.py --file_name generate_sentences.txt` to calculate the metrics.


5. When fine-tuning the model on covidialogue, download the covid dataset from [here](https://drive.google.com/file/d/1i7nxb4dvwKV6zW8pUQFcpEgVs3Wyu6J7/view?usp=sharing) and make a preprocess on it,
then you can run `python bert_gpt_train.py --decoder_model ${the path of the pretrained model}`, and run
again the instruction shown by 3-4.



### Requirements
chinese-gpt==0.1.3
pytorch==1.4.0
fire
tqdm
numpy
allennlp==0.9.0
pytorch-pretrained-bert==0.6.2
nlt


### Pre-trained models
We provide the pre-trained models here:

**BERT-GPT trained on MedDialog** ([bertGPT pretrained model.pth7](https://drive.google.com/file/d/1alyU4wEClpjj2-kGl45xxUal0dHZGZhI/view?usp=sharing))
