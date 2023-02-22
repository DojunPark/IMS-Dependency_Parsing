import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from parser_models import oracle, Parser


def reading(txt):
    lines = txt.strip().split('\n')
    buffers = []
    buffer = [('[ROOT]', '[ROOT]', 0)]

    for line in lines:
        if line.startswith('#'):
            continue
        columns = line.split('\t')
        if not columns[0].isdigit():
            buffers.append(buffer)
            buffer = [('[ROOT]', '[ROOT]', 0)]
        else:
            buffer.append((columns[1], columns[3], int(columns[6]) if columns[6].isdigit() else columns[6]))
    buffers.append(buffer)
    return buffers


def writing(heads, rfile_name, wfile_name):
    with open(rfile_name, encoding='utf-8') as f:
        en = f.read()
    
    ls_en = en.split('\n\n')
    ls_en = ls_en[:-1]

    post_ls_en = []
    for x, y in zip(heads, ls_en):
        ls_y = y.split('\n')

        post_ls_y = []
        for elem_x, elem_y in zip(x, ls_y):
            ls_elem_y = elem_y.split('\t')
            ls_elem_y[6] = str(elem_x)
            elem_y = '\t'.join(ls_elem_y)
            post_ls_y.append(elem_y)
        post_y = '\n'.join(post_ls_y)
        post_ls_en.append(post_y)
        
    post_ls_en.append('')
    post_en = '\n\n'.join(post_ls_en)
    
    with open(wfile_name, 'w', encoding='utf-8') as f:
        f.write(post_en)


def construct_word_vocab_dict(sents):
    word_vocab = {'[PAD]': 0, '[UNK]': 1}
    pos_vocab = {'[PAD]': 0}
    
    for sent in sents:
        for word, pos, _ in sent:
            word_vocab.setdefault(word, len(word_vocab))
            pos_vocab.setdefault(pos, len(pos_vocab))
    return word_vocab, pos_vocab

        
def predict_dependencies(parser, test_sentences):
    sent_predictons = []
    for sentence in test_sentences:
        words, tags, _ = zip(*sentence)
        pred_heads = parser.predict(words, tags)
        sent_predictons.append(pred_heads[1:])
    return sent_predictons


class DependencyParsingDataset(Dataset):
    def __init__(self, sents, parser):
        self.features = []
        self.labels = []
        for sent in sents:
            words, tags, gold_heads = zip(*sent)
            word_ids = [parser.word_vocab.get(w, 0) for w in words]
            tag_ids = [parser.pos_vocab[t] for t in tags]

            for config, action in oracle(gold_heads):
                features = parser.featurize(word_ids, tag_ids, config)
                label = action
                self.features.append(features)
                self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_uas(parser, sents):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.model.to('cpu')
    total = 0
    correct = 0
    for sent in sents:
        words, tags, gold_heads = zip(*sent)
        pred_heads = parser.predict(words, tags)
        for gold, pred in zip(gold_heads[1:], pred_heads[1:]):
            total += 1
            correct += int(gold == pred)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return correct / total


def train_dependency_parser(train_data, dev_data, test, test_file, save_name, n_epochs=1, batch_size=100, lr=1e-2):
    vocab_words, vocab_tags = construct_word_vocab_dict(train_data)
    parser = Parser(vocab_words, vocab_tags)
    dataset = DependencyParsingDataset(train_data, parser)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser.model.to(device)
        if epoch == 0:
            print('The current training is running on: ', device)

        running_loss = 0
        n_examples = 0
        with tqdm(total=sum(2*len(s)-1 for s in train_data)) as pbar:
            for bx, by in dataloader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                output = parser.model.forward(bx)
                loss = F.cross_entropy(output, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_examples += 1
                pbar.set_postfix(loss=running_loss/n_examples)
                pbar.update(len(bx))
        print(f'{epoch+1}th epoch >>> ', end='')
        print('dev_uas: {:.4f}'.format(compute_uas(parser, dev_data)))
        if not os.path.exists('./model'):
            os.makedirs('./model')
        torch.save(parser.model, 'model/parser_{}.pt'.format(str(epoch+1)))

        # save the data
        test_heads = predict_dependencies(parser, test)
        writing(test_heads, test_file, save_name+'_'+str(epoch))
        print('data saved')


    return parser   



