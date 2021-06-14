from train_v4 import Net


x = torch.zeros((vocab_size, 1))
x = torch.randn((vocab_size, 1))
x[char_to_index['<']] = 1
x = x.reshape((-1, 1, vocab_size)).to(device)

h_0 = torch.zeros(2, 1, 1024).to(device)
c_0 = torch.zeros(2, 1, 1024).to(device)

rnn = nn.LSTM(29, 1024, 1, batch_first = True).to(device)

output, (hn, cn) = rnn(x, (h_0, c_0))

fc = nn.Linear(1024, 29).to(device)

output = fc(output)

output, (hn, cn) = net(x, h = h_0, c = c_0, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'))

print(net)

outputs = net(x = x, h = None, c = None, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'))

idx = outputs.cpu().data.numpy().argmax()

x = torch.zeros((1, vocab_size)).to(device)
x[0][idx] = 1
x = x.unsqueeze(0)

output, (hn, cn) = rnn(x, (hn, cn))



tmp = torch.tensor([[[1, 3, 13, 14, 15]]]).squeeze(0).float()
tmp = torch.tensor([[[1, 3, 13, 14, 15]]])
tmp = tmp.view(-1)
tmp.reshape(1, 1, 5)
tmp.shape

torch.multinomial(tmp, 1)

mod[5][-1]



out = pack_padded_sequence(lines_encoded[:5], lengths = [40, 38, 37, 34, 32], batch_first = True)

ttt = torch.tensor([1, 2, 3])
ttt = ttt[:-1]


aaa = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2])]
torch.nn.utils.rnn.pad_sequence(aaa, batch_first = True, padding_value = 0)



batch


b = batch[:5]
b = torch.nn.utils.rnn.pad_sequence(b, batch_first = True, padding_value = 0)

can = [torch.eye(input_size)[x] for x in batch]
can = torch.tensor(can)





len(batch[1][0, 1][:41, :38])

sampl = dataset[1].tolist()
sampl = batch[1][1].tolist()
result_str = ''.join([index_to_char[c] for c in sampl])
print(result_str)








X = torch.nn.utils.rnn.pack_padded_sequence(batch[0].to(device), batch[2].to(device), batch_first = True, enforce_sorted = False)

lengths = batch[2]

max_len = batch[3]

mod_1 = torch.nn.RNN(29, 29, num_layers = 1, batch_first=True).to(device)

out = mod_1(X)

out = torch.nn.utils.rnn.pad_packed_sequence(out[0], batch_first = True)

mod_2 = torch.nn.Linear(29, 29, bias=True).to(device)

out = mod_2(out[0])
x = torch.zeros((vocab_size, 1))
x[char_to_index['<']] = 1
x = x.reshape((-1, 1, vocab_size)).to(device)
out = lstm_model(x, torch.as_tensor([len(x)], dtype = torch.int64, device = 'cpu'))
out = lstm_model(out, torch.as_tensor([len(out)], dtype = torch.int64, device = 'cpu'))







outputs = rnn_model(x, torch.as_tensor([len(x)], dtype = torch.int64, device = 'cpu'))
result = outputs.cpu().data.numpy().argmax()



X, Y, lengths = batch[0].to(device), batch[1].to(device), batch[2].to(device)
# X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first = True, enforce_sorted = False)
outputs = lstm_model(X, lengths)

result = outputs.cpu().data.numpy().argmax(axis=2)
result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
print(result_str)

true = ''.join([index_to_char[c] for c in Y.squeeze().tolist()])
print(true)

outputs = lstm_model(x, torch.as_tensor([2], dtype = torch.int64, device = 'cpu'))