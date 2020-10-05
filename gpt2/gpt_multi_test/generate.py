import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


PAD = '[PAD]'
pad_id = 0

def create_model(args, vocab_size):
    """

    :param args:
    :param vocab_size:字典大小
    :return:
    """
    if args.pretrained_model:  # 如果指定了预训练的GPT2模型
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=50, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialogue_model_path', default='dialogue_model/med_model_epoch1', type=str, required=False, help='对话模型路径')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=100, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=10, help="dialogue history的最大长度")
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
    

def main():
    args = set_interact_args()
    logger = create_logger(args)

    device = torch.device("cuda:7")
    data_num = 11
    
    print ("Device: ", device)
    print ("Data_num: ", data_num)
    input_file_name = "data/test_med_multi/test_" + str(data_num) + ".txt"
    output_file_name = "generation/test_med_multi/test_" + str(data_num) + ".json"
    
    logger.info('using device:{}'.format(device))
    
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
    model.to(device)
    model.eval()
    with open(input_file_name, "rb") as f:
        input_data = f.read().decode("utf-8")
    f.close()
    input_data = input_data.split("\n\n")
    
    res = []
    
    for dialogs in tqdm(input_data):
        history = []
        utterances = dialogs.split("\n")
        total = int(len(utterances) / 2)
        for index in range(total):
            utterance = utterances[2 * index]
            text = utterance
            history.append(tokenizer.encode(text))
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            curr_input_tensor = torch.tensor(input_ids).long().to(device)
            generated = []
            # 最多生成max_len个token
            for _ in range(args.max_len):
                over = len(curr_input_tensor) - 300
                if over > 0:
                    curr_input_tensor = curr_input_tensor[over: ]
                outputs = model(input_ids=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            text = tokenizer.convert_ids_to_tokens(generated)
            text = "".join(text)
            target_utt = utterances[2 * index + 1]
            res.append([text, target_utt])
            
            history.append(tokenizer.encode(target_utt))
    with open(output_file_name, "w") as f:
        json.dump(res, f, ensure_ascii = False, indent = 4)
    f.close()
    print ("data#" + str(data_num) + " finished.")
    

if __name__ == '__main__':
    main()
