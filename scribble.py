from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

mod_1 = torch.nn.RNN(29, 29, num_layers = 1, batch_first=True).to(device)

out = mod_1(X)

out = torch.nn.utils.rnn.pad_packed_sequence(out[0], batch_first = True)

mod_2 = torch.nn.Linear(29, 29, bias=True).to(device)

out = mod_2(out[0])

net(X, lengths)