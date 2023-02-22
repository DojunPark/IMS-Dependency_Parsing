'''
Arguments example:
python run.py --train_file data/english/train/wsj_train.only-projective.conll06 --dev_file data/english/dev/wsj_dev.conll06.gold --test_file data/english/test/wsj_test.conll06.blind --save model/wsj_test.conll06.pred --epoch 5
python run.py --train_file data/german/train/tiger-2.2.train.only-projective.conll06 --dev_file data/german/dev/tiger-2.2.dev.conll06.gold --test_file data/german/test/tiger-2.2.test.conll06.blind --save model/tiger-2.2.test.conll06.pred --epoch 2
'''

# import modules 
from utils import reading, writing, construct_word_vocab_dict, predict_dependencies, train_dependency_parser, compute_uas
import argparse

# Create an argument parser
argparser = argparse.ArgumentParser()

# Add arguments
argparser.add_argument("--train_file", help="Path to the training file")
argparser.add_argument("--dev_file", help="Path to the development file")
argparser.add_argument("--test_file", help="Path to the test file")
argparser.add_argument("--save", help="Path to saving file")
argparser.add_argument("--epoch", help="The desired epoch number")

# Parse the arguments
args = argparser.parse_args()

# Access the arguments
train_name = args.train_file
dev_name = args.dev_file
test_name = args.test_file
epoch = args.epoch
epochs = int(epoch)
wfname = args.save
save_name = test_name.split('/')[-1]
save_name = save_name.replace('blind', 'pred')


with open(train_name, 'r', encoding='utf-8') as f:
    train_source = f.read()
with open(dev_name, 'r', encoding='utf-8') as f:
    dev_source = f.read()    
with open(test_name, 'r', encoding='utf-8') as f:
    test_source = f.read()

train = reading(train_source)
dev = reading(dev_source)
test = reading(test_source)


# train model
save_name = 'model/wsj_test.conll06.pred_'
parser = train_dependency_parser(train, dev, test, test_name, save_name, n_epochs=epochs)

