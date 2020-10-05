import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams
from collections import defaultdict
import numpy as np


if __name__ == "__main__":
    total_data = 12
    print ("Loading input data.")
    input_token = []
    total_input = 0
    for i in range(total_data):
        input_file_name = "data/test_med_multi/test_" + str(i) + ".txt"
        with open(input_file_name, "rb") as f:
            temp_input = f.read().decode("utf-8")
        f.close()
        temp_input = temp_input.split("\n\n")
        for dialog in temp_input:
            utterances = dialog.split("\n")
            total_utt = len(utterances) // 2
            history = ""
            for index in range(total_utt):
                text_user = utterances[2 * index]
                text_response = utterances[2 * index + 1]
                if index > 0:
                    history += "[SEP]"
                history += text_user
                input_token.append(history)
                history += "[SEP]" + text_response
                total_input += 1
        print ("Input_data#" + str(i) + " loaded.")

    print ("Loading data.")
    pred_token = []
    target_token = []
    total_output = 0
    
    for index in range(total_data):
        input_file_name = "generation/test_med_multi/test_" + str(index) + ".json"
        with open(input_file_name, "r") as f:
            temp_res = json.load(f)
        f.close()
        for pairs in temp_res:
            pred_token.append(pairs[0])
            target_token.append(pairs[1])
            total_output += 1
        print ("Data#" + str(index) + " loaded.")
        
    if (total_input == total_output):
        print ("Matched successfully.")
        print ("All data loaded.")
        length = len(pred_token)
        print ("Test number: ", length)
        f_output = open("generation/DialogGPT_generation.txt", "w")
        for index in range(length):
            f_output.write("Case#"+ str(index) + "\n")
            f_output.write("Input: " + input_token[index] + "\n")
            f_output.write("Target: " + target_token[index] + "\n")
            f_output.write("Prediction: " + pred_token[index] + "\n")
        f_output.close()
    else:
        print ("Failed!")
        print ("Total input: ", total_input)
        print ("Total output : ", total_output)
    

    
