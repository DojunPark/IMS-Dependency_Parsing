import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralClassifier(nn.Module):
    
    def __init__(self, num_words, num_tokens, word_dim=100, pos_dim=100, hidden_dim=100):
        super().__init__()
        self.word_embedding = nn.Embedding(num_words, word_dim)
        self.pos_embedding = nn.Embedding(num_tokens, pos_dim)
        self.linear1 = nn.Sequential(nn.Linear(word_dim*9 + pos_dim*9, hidden_dim*6), nn.ReLU())
        self.linear2 = nn.Linear(hidden_dim*6, 3)    
    
    def forward(self, x):
        embedded = [
            self.word_embedding(x[..., 0]),
            self.word_embedding(x[..., 1]),
            self.word_embedding(x[..., 2]),
            self.word_embedding(x[..., 3]),
            self.word_embedding(x[..., 4]),
            self.word_embedding(x[..., 5]),
            self.word_embedding(x[..., 6]),
            self.word_embedding(x[..., 7]),
            self.word_embedding(x[..., 8]),
            self.pos_embedding(x[..., 9]),
            self.pos_embedding(x[..., 10]),
            self.pos_embedding(x[..., 11]),
            self.pos_embedding(x[..., 12]),
            self.pos_embedding(x[..., 13]),
            self.pos_embedding(x[..., 14]),
            self.pos_embedding(x[..., 15]),
            self.pos_embedding(x[..., 16]),
            self.pos_embedding(x[..., 17])]
        
        embedded = torch.cat(embedded, -1)
        y = self.linear1(embedded)
        y = self.linear2(y)
        return y
 

class Parser(object):
    
    SHIFT, L_ARC, R_ARC = (0, 1, 2)
    
    def __init__(self, word_vocab, pos_vocab):
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.model = NeuralClassifier(len(word_vocab), len(pos_vocab))
    
    @staticmethod
    def initial_config(num_words):
        return 0, [], [0] * num_words
    
    @staticmethod
    def valid_moves(config):
        pos, stack, heads = config
        moves = []
        if pos < len(heads):
            moves.append(Parser.SHIFT)
        if len(stack) >= 2:
            moves.append(Parser.L_ARC)
            moves.append(Parser.R_ARC)
        return moves
    
    @staticmethod
    def next_config(config, move):
        pos, stack, heads = config
        stack = list(stack)    
        if move == Parser.SHIFT:
            stack.append(pos)
            pos += 1
        else:
            heads = list(heads)
            s1 = stack.pop()
            s2 = stack.pop()
            if move == Parser.L_ARC:
                heads[s2] = s1
                stack.append(s1)
            if move == Parser.R_ARC:
                heads[s1] = s2
                stack.append(s2)               
        return pos, stack, heads
    
    @staticmethod
    def is_final_config(config):
        pos, stack, heads = config
        return pos == len(heads) and len(stack) == 1


    
    def featurize(self, word_indices, pos_indices, config):
        i, stack, heads = config
        x = torch.zeros(18, dtype=torch.long)
        x[0] = word_indices[i] if i < len(word_indices) else 0
        x[1] = word_indices[stack[-1]] if len(stack) >= 1 else 0
        x[2] = word_indices[stack[-2]] if len(stack) >= 2 else 0
        x[3] = word_indices[stack[-3]] if len(stack) >= 3 else 0
        x[4] = word_indices[stack[-4]] if len(stack) >= 4 else 0
        x[5] = word_indices[stack[-5]] if len(stack) >= 5 else 0
        x[6] = word_indices[stack[-6]] if len(stack) >= 6 else 0
        x[7] = word_indices[stack[-7]] if len(stack) >= 7 else 0
        x[8] = word_indices[stack[-8]] if len(stack) >= 8 else 0
        x[9] = pos_indices[i] if i < len(pos_indices) else 0
        x[10] = pos_indices[stack[-1]] if len(stack) >= 1 else 0
        x[11] = pos_indices[stack[-2]] if len(stack) >= 2 else 0
        x[12] = pos_indices[stack[-3]] if len(stack) >= 3 else 0
        x[13] = pos_indices[stack[-4]] if len(stack) >= 4 else 0
        x[14] = pos_indices[stack[-5]] if len(stack) >= 5 else 0
        x[15] = pos_indices[stack[-6]] if len(stack) >= 6 else 0
        x[16] = pos_indices[stack[-7]] if len(stack) >= 7 else 0
        x[17] = pos_indices[stack[-8]] if len(stack) >= 8 else 0
        return x

    
    def predict(self, ls_word, ls_pos):
        ls_word = [self.word_vocab.get(w, 0) for w in ls_word]
        ls_pos = [self.pos_vocab.get(t, 0) for t in ls_pos]
        config = self.initial_config(len(ls_word))
        valid_moves = self.valid_moves(config)
        while valid_moves:
            features = self.featurize(ls_word, ls_pos, config)
            with torch.no_grad():
                scores = self.model.forward(features)
                
            best_score, pred_move = float('-inf'), None
            for move in valid_moves:
                if scores[move] > best_score:
                    best_score, pred_move = scores[move], move
                    
            config = self.next_config(config, pred_move)
            valid_moves = self.valid_moves(config)
        i, stack, pred_heads = config
        return pred_heads
    
    
def oracle(ls_heads):
    remaining_cnt = [0] * len(ls_heads)
    for head in ls_heads:
        remaining_cnt[head] += 1

    config = Parser.initial_config(len(ls_heads))
    while not Parser.is_final_config(config):
        pos, stack, heads = config
        if len(stack) >= 2:
            s1, s2 = stack[-1], stack[-2]
            if ls_heads[s2] == s1 and remaining_cnt[s2] == 0:
                move = Parser.L_ARC
                yield config, move
                config = Parser.next_config(config, move)
                remaining_cnt[s1] -= 1
                continue
            if ls_heads[s1] == s2 and remaining_cnt[s1] == 0:
                move = Parser.R_ARC
                yield config, move
                config = Parser.next_config(config, move)
                remaining_cnt[s2] -= 1
                continue
        move = Parser.SHIFT
        yield config, move
        config = Parser.next_config(config, move)