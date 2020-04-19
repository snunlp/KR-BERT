import argparse
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from pretrained.tokenization_ranked import FullTokenizer as KBertRankedTokenizer
from transformers import BertTokenizer as BertTokenizer
from model.net import SentenceClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--dataset', default='test', help="name of the data in --data_dir to be evaluate")
parser.add_argument('--pretrained_config', default=None, required=False, type=str)
parser.add_argument('--subchar', default='False', choices=['False', 'True'], required=True)
parser.add_argument('--tokenizer', default='ranked', choices=['ranked','bert'], required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    ptr_dir = Path('pretrained')
    data_dir = Path(args.data_dir)
    model_dir = Path('checkpoints')

    # pretrained config
    args.pretrained_config = 'subchar12367' if args.subchar == 'True' else 'char16424'
    args.pretrained_config = args.pretrained_config + '_' + args.tokenizer
    print('[CONFIG] config_{}.json'.format(args.pretrained_config))

    ptr_config = Config(ptr_dir / 'config_{}.json'.format(args.pretrained_config))
    data_config = Config(data_dir / 'config.json')
    model_config = Config('finetuning_config.json')

    # vocab
    vocab = pickle.load(open(ptr_config.vocab, mode='rb'))

    # tokenizer
    if args.tokenizer == 'ranked':
        print('[RANKED TOKENIZER]')
        ptr_tokenizer = KBertRankedTokenizer(ptr_config.tokenizer, do_lower_case=False)
    else:
        ptr_tokenizer = BertTokenizer.from_pretrained(ptr_config.tokenizer, do_lower_case=False)
        print('[BERT TOKENIZER]')
    pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
    preprocessor = PreProcessor(vocab=vocab, split_fn=ptr_tokenizer.tokenize, pad_fn=pad_sequence,subchar =args.subchar)


    # model (restore)
    checkpoint_manager = CheckpointManager(model_dir)
    checkpoint = checkpoint_manager.load_checkpoint('best_snu_{}.tar'.format(args.pretrained_config))

    config = BertConfig(ptr_config.config)
    model = SentenceClassifier(config, num_classes=model_config.num_classes, vocab=preprocessor.vocab)
    model.load_state_dict(checkpoint['model_state_dict'])

    # evaluation
    filepath = getattr(data_config, args.dataset)
    ds = Corpus(filepath, preprocessor.preprocess)
    dl = DataLoader(ds, batch_size=model_config.batch_size, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    summary_manager = SummaryManager(model_dir)
    summary = evaluate(model, dl, {'loss': nn.CrossEntropyLoss(), 'acc': acc}, device)

    summary_manager.load('summary_snu_{}.json'.format(args.pretrained_config))
    summary_manager.update({'{}'.format(args.dataset): summary})
    summary_manager.save('summary_snu_{}.json'.format(args.pretrained_config))

    print('loss: {:.3f}, acc: {:.2%}'.format(summary['loss'], summary['acc']))
