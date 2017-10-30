import time
import os
from functools import partial
import fileinput

import numpy as np

from _main import sq
from _main import mle
from _main import decode
from _decode import decode_lm

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.SeqModel.default_opt(),
                     'train': sq.default_training_opt()}
    parser = sq.get_common_argparser('main_ngram.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt = all_opt

    def data_fn():
        dpath = partial(os.path.join, opt['data_dir'])
        vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
        data_fn = partial(
            sq.read_ngram_data, in_vocab=vocab, out_vocab=vocab, normalized=True)
        data = [data_fn(sq.read_lines(dpath(f), token_split=' ', part_split='\t'))
                for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]
        batch_iter = partial(sq.ngram_batch_iter, batch_size=opt['batch_size'])
        return data, batch_iter, (vocab, vocab)

    mle(opt, model_opt, train_opt, logger, data_fn, sq.SeqModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
