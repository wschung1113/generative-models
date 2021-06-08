import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

# from tensorflow.keras.utils import to_categorical

df_orig = pd.read_csv('train.csv')

df = df_orig.iloc[:500000, ]

n = df.shape[0]

lines = []

for i in range(n):
    line = str(df.iloc[i, ][0])
    if len(line) > 0:
        lines.append(line)

for i in range(10):
    print(lines[i])

max_len = len(lines[0])
for i in range(1, len(lines)):
    max_len = max(max_len, len(lines[i]))
print(max_len) # 가장 긴 SMILES 길이

text = ' '.join(lines)
print('문자열의 길이 또는 총 글자의 개수: %d' % len(text))

tmp = list(set(text))
tmp.append('<') # SoS token
tmp.append('>') # EoS token
char_vocab = sorted(tmp)
vocab_size = len(char_vocab)
print ('글자 집합의 크기 : {}'.format(vocab_size))
print(char_vocab)

char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
print(char_to_index)

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key
# print(index_to_char)

train_X = []
train_y = []

for i in range(n): 
    X_sample = lines[i][:-1]
    X_encoded = [10] + [char_to_index[c] for c in X_sample] + [12] # 정수 인코딩 # SoS, EoS 토큰 부착
    train_X.append(X_encoded)

    y_sample = lines[i][1:]
    y_encoded = [10] + [char_to_index[c] for c in y_sample] + [12] # 정수 인코딩 # SoS, EoS 토큰 부착
    train_y.append(y_encoded)

# # 정수 인코딩 SMILES 예시
# print(train_X[0])
# print(train_y[0])

# 원-핫 인코딩 하기
train_X_onehot = []
# train_y_onehot = []

for i in range(n):
    t_X = train_X[i]
    a_X = torch.zeros((len(t_X), vocab_size))
    a_X[range(len(t_X)), t_X] = 1
    train_X_onehot.append(a_X)

    # t_y = train_y[i]
    # a_y = torch.zeros((len(t_y), vocab_size))
    # a_y[range(len(t_y)), t_y] = 1
    # train_y_onehot.append(a_y)

print(train_X_onehot[0]) # 원-핫 임베딩 된 SMILES 벡터
print(train_X_onehot[0].shape)
# print(train_X_onehot[0].shape)
# print(len(train_y[0])) # y는 원-핫 변환 안 함
# train_y = torch.LongTensor(train_y[0])

# print(train_y_onehot[0:3]) # 원-핫 임베딩 된 SMILES 벡터

# 모델 구현
input_size = vocab_size # 각 시점 입력의 크기 = 원-핫 벡터의 길이
hidden_size = vocab_size # hidden_size 역시 원-핫 벡터의 길이
output_size = vocab_size # 출력 사이즈 역시 vocab_size
learning_rate = 0.1

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True) # RNN 셀 구현
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True) # 출력층 구현

    def forward(self, x): # 구현한 RNN 셀과 출력층을 연결
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(input_size, hidden_size, output_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

# # 출력 크기 확인
# idx = 0
# X = torch.unsqueeze(train_X_onehot[idx], 0)
# Y = torch.LongTensor(train_y[idx])
# outputs = net(X)
# print(outputs)
# print(outputs.shape) # 3차원 텐서 # 샘플마다 2차원의 길이가 상이 # (1, x, 28)
# loss = criterion(outputs.view(-1, input_size), Y.view(-1))
# loss.backward() # 기울기 계산
# optimizer.step()
# result = outputs.data.numpy().argmax(axis=2) # 최종 예측값인 각 time-step 별 5차원 벡터에 대해서 가장 높은 값의 인덱스를 선택
# result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
# print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", Y, "prediction str: ", result_str)

epoch = 10
for i in range(epoch):
    optimizer.zero_grad()
    for j in range(n): # for every sample
        X = torch.unsqueeze(train_X_onehot[j], 0)
        Y = torch.LongTensor(train_y[j])
        outputs = net(X)
        loss = criterion(outputs.view(-1, input_size), Y.view(-1))
        loss.backward() # 기울기 계산
        optimizer.step() # 아까 optimizer 선언 시 넣어둔 파라미터 업데이트
    

    # 아래 세 줄은 모델이 실제 어떻게 예측했는지를 확인하기 위한 코드.
    result = outputs.data.numpy().argmax(axis=2) # 최종 예측값인 각 time-step 별 5차원 벡터에 대해서 가장 높은 값의 인덱스를 선택
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", Y, "prediction str: ", result_str)

# # 체크포인트 생성
# # 구현하는 .py 짜기