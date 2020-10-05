#### How to Run

1. When training, run `gpt_train.py`. Notice that, if you run the training file first time, please use `python gpt_train.py --raw` to tokenize.
2. When calculating the perplexity, run `gpt2_ppl.py`.
3. When testing, run `gpt2_test.py` for a single process and run "gpt_multi_test/generate.py" for parallel computing.

**Notice that, both the training and testing file should be groups of dialogue text, which is split by a blank line.**

> For example, the train.txt should be like:
> What are you doing?
> Chating with you.
> I see.
>
> Hi, are you okay?
> Yes, I am fine.
>
> ......

We use make_test.py to implement this.

When using pretrain, please download the pretrain model and put it under the file "dialogue_model".



#### Requirements
transformers==2.1.1
pytorch==1.4.0
sklearn
tqdm
numpy
scipy==1.2.1



#### Acknowledge

This GPT2 model code have developed based on the [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat) and [DialoGPT](https://github.com/microsoft/DialoGPT).

