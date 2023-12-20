import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, BertModel, BertConfig, get_linear_schedule_with_warmup

from tqdm import tqdm

import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util

#os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

class transforers_model(nn.Module):
    def __init__(self):
        super().__init__()

        # BERT 모델의 설정 구성
        encoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8
        )

        # 인코더 설정
        self.encoder = BertModel(encoder_config)

        # 디코더 설정
        decoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8
        )
        decoder_config.is_decoder = True
        self.decoder = BertModel(decoder_config)

        # 선형 레이어 설정
        self.linear = nn.Linear(512, 21128, bias=False)

    # forward 메서드 정의
    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        # 인코더의 hidden states 계산
        encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        # 디코더의 출력 계산
        out, _ = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        
        # 선형 레이어 통과
        out = self.linear(out)
        return out

# train_model 함수 정의
def train_model(
    epochs=15,  # 전체 데이터셋을 몇 번 반복하여 학습할 것인지 지정하는 변수
    num_gradients_accumulation=4,  # 그래디언트 업데이트를 몇 번에 한 번씩 적용할 것인지 지정하는 변수
    batch_size=64,  # 각 미니배치의 크기를 나타내는 변수
    gpu_id=0,  # GPU를 사용할 경우 선택할 GPU의 ID를 지정하는 변수
    lr=1e-4,  # 학습률(learning rate)로, 모델 가중치를 업데이트하는 정도를 나타내는 변수
    load_dir='decoder_model'  # 모델의 저장 및 로드할 디렉토리 경로를 지정하는 변수
    ):
        
    # 모델이 GPU에 있도록 함
    device = torch.device("cuda:5")

    # 모델 로드
    model = transforers_model()
    # PyTorch에서 제공하는 모델 병렬 처리를 위한 래퍼(wrapper) 클래스입니다. 
    # 이 클래스를 사용하면 모델을 여러 GPU에 병렬적으로 실행할 수 있습니다. 
    # 주어진 모델을 nn.DataParallel로 래핑하면, 모델이 각 GPU에서 동시에 병렬로 처리됩니다.
    model = nn.DataParallel(model, device_ids=[5,6,7])
    model = model.to(device)

    # 훈련 및 검증 데이터 로드
    train_data = torch.load("train_data.pth")
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    
    val_data = torch.load("validate_data.pth")
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)

    # 옵티마이저 설정
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps // 10,
        num_training_steps=num_train_optimization_steps
    )

    # 훈련 시작
    update_count = 0
    f = open("valid_loss.txt", "w")
    start = time.time()
    print('훈련 시작....')
    for epoch in range(epochs):
        # 훈련 단계
        model.train()
        losses = 0
        times = 0
        for batch in tqdm(train_dataloader):
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()
            times += 1
            update_count += 1
            max_grad_norm = 1.0

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end = time.time()
        print('-'*20 + '에포크' + str(epoch) + '-'*20)
        print('시간:' + str(end - start))
        print('손실:' + str(losses / times))
        start = end

        # 검증 단계
        model.eval()

        perplexity = 0
        temp_loss = 0
        batch_count = 0
        print('퍼플렉서티 계산 시작....')

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = [item.to(device) for item in batch]

                encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
                logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

                temp_loss += loss.item()
                perplexity += np.exp(loss.item())

                batch_count += 1

        print('검증 퍼플렉서티:' + str(perplexity / batch_count))
        print("검증 손실:" + str(temp_loss / batch_count))
        
        f.write('-'*20 + f"에포크 {epoch}" + '-'*20 + '\n')
        f.write(f"퍼플렉서티: {str(perplexity / batch_count)}" + "\n")
        f.write(f"손실: {str(temp_loss / batch_count)}" + "\n\n")
        
        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)

        torch.save(model.module.state_dict(), os.path.join(os.path.abspath('.'), load_dir, str(epoch) + "model.pth"))
    f.close()

if __name__ == '__main__':
    fire.Fire(train_model)
