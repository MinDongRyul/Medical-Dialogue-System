import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams
from collections import defaultdict
import numpy as np

def get_metrics(pred, target):
    turns = len(target)
    bleu_2 = 0
    bleu_4 = 0
    meteor = 0
    nist_2 = 0
    nist_4 = 0
    for index in range(turns):
        pred_utt = pred[index]
        target_utt = target[index]
        min_len = min(len(pred_utt), len(target_utt))
        lens = min(min_len, 4)
        if lens == 0:
            continue
        if lens >= 4:
            bleu_4_utt = sentence_bleu([target_utt], pred_utt, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function = SmoothingFunction().method1)
            nist_4_utt = sentence_nist([target_utt], pred_utt, 4)
        else:
            bleu_4_utt = 0
            nist_4_utt = 0
        if lens >= 2:
            bleu_2_utt = sentence_bleu([target_utt], pred_utt, weights = (0.5, 0.5), smoothing_function = SmoothingFunction().method1)
            nist_2_utt = sentence_nist([target_utt], pred_utt, 2)
        else:
            bleu_2_utt = 0
            nist_2_utt = 0
            
        bleu_2 += bleu_2_utt
        bleu_4 += bleu_4_utt
        meteor += meteor_score([" ".join(target_utt)], " ".join(pred_utt))
        nist_2 += nist_2_utt
        nist_4 += nist_4_utt
        
    bleu_2 /= turns
    bleu_4 /= turns
    meteor /= turns
    nist_2 /= turns
    nist_4 /= turns
    return bleu_2, bleu_4, meteor, nist_2, nist_4

def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

if __name__ == "__main__":

    print ("Loading data.")
    total_data = 12
    pred_token = []
    target_token = []
    
    for index in range(total_data):
        input_file_name = "generation/test_med_multi/test_" + str(index) + ".json"
        with open(input_file_name, "r") as f:
            temp_res = json.load(f)
        f.close()
        for pairs in temp_res:
            pred_token.append(pairs[0])
            target_token.append(pairs[1])
        print ("Data#" + str(index) + " loaded.")
        
    print ("All data loaded.")
    length = len(pred_token)
    print ("Test number: ", length)
    
    ave_len = 0
    pred = []
    for index in range(length):
        pred.append(" ".join(pred_token[index]))
        ave_len += len(pred_token[index])
    
    
#    target_token = target_token[0:2]
    bleu_2, bleu_4, meteor, nist_2, nist_4 = get_metrics(pred_token, target_token)
    entropy, dist = cal_entropy(pred)
    ave_len /= length


    print ("Bleu_2: ", bleu_2)
    print ("Bleu_4: ", bleu_4)
    print ("Meteor: ", meteor)
    print ("Nist_2: ", nist_2)
    print ("Nist_4: ", nist_4)
    print ("Dist_1: ", dist[0])
    print ("Dist_2: ", dist[1])
    print ("Entropy_4: ", entropy[3])
    print ("Length: ", ave_len)
        
    
