import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch import from_numpy
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using CUDA!")
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda()
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy


def get_bert(bert_model, bert_do_lower_case, model_path=None):
    with tqdm(desc='Loading model', total=4 if model_path else 2) as pbar:
        #  from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel, XLMRobertaTokenizer, XLMRobertaModel, DistilBertTokenizer, DistilBertModel
        #  from transformers import ElectraModel, ElectraTokenizer
        #
        #  if model_path is None:
        #      tokenizer =  XLMRobertaTokenizer.from_pretrained(bert_model, do_lower_case = bert_do_lower_case)
        #      model = XLMRobertaModel.from_pretrained(bert_model)
        #  else:
        #      tokenizer =  BertTokenizer.from_pretrained(bert_model, do_lower_case = bert_do_lower_case)
        #      model = BertModel.from_pretrained(bert_model)
        #      state_dict_1 = torch.load(model_path)
        #      '''
        #      state_dict_2 = {}
        #      for k, v in state_dict_1.items():
        #        if 'bert' in k:
        #          k = k.split(".")[1:]
        #          k = ".".join(k)
        #        state_dict_2[k] = v
        #      '''
        #      model.load_state_dict(state_dict_1, strict=True)

        from transformers import BertTokenizer, BertModel
        model = BertModel.from_pretrained(bert_model)
        pbar.update(1)

        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
        pbar.update(1)

        if model_path is not None:
            state_dict_1 = torch.load(model_path)['state_dict']
            pbar.update(1)

            state_dict_2 = {}
            for k, v in state_dict_1.items():
                if 'bert' in k:
                    k = k.split(".")[1:]
                    k = ".".join(k)
                state_dict_2[k] = v
            model.load_state_dict(state_dict_2, strict=True)
            pbar.update(1)

    return tokenizer, model


class WordLevelBert(nn.Module):
    """
    Runs BERT on sentences but only keeps the last subword embedding for
    each word.
    """

    def __init__(self, model, do_lower_case, model_path=None):
        super().__init__()
        self.bert_tokenizer, self.bert = get_bert(model, do_lower_case, model_path)
        #  self.dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings

        if use_cuda:
            self.cuda()

    def forward(self, sentences, include_clssep = True):
        batch_size = 128
        ann_full = None
        for i in range(0, len(sentences), batch_size):
            ann = self.annotate(sentences[i:i+batch_size],
                                include_clssep = include_clssep)
            if ann_full is None:
                ann_full = ann
            else:
                ann_full = torch.cat((ann_full, ann), dim = 0)
        return ann_full

    def annotate(self, sentences, include_clssep=True):
        return self.annotate_bert(sentences, include_clssep)

    def annotate_bert(self, sentences, include_clssep=True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (packed_len, dim).
            packed_len is the total number of words, plus 2 for each sentence
            for [CLS] and [SEP].
        """
        if include_clssep:
            packed_len = sum([(len(s) + 2) for s in sentences])
        else:
            packed_len = sum([len(s) for s in sentences])

        # Each row is the token ids for a sentence, padded with zeros.
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for the last subword for each word.
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        max_sent = 0
        for s_num, sentence in enumerate(sentences):
            tokens = []
            end_mask = []
            tokens.append("[CLS]")
            end_mask.append(int(include_clssep))
            for word in sentence:
                word_tokens = self.bert_tokenizer.tokenize(word)
                assert len(word_tokens) > 0, \
                    "Unknown word: {} in {}".format(word, sentence)
                for _ in range(len(word_tokens)):
                    end_mask.append(0)
                end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("[SEP]")
            end_mask.append(int(include_clssep))
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(end_mask)] = end_mask
            max_sent = max(max_sent, len(input_ids))
        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        all_end_mask = all_end_mask[:, :max_sent]
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask))

        # all_input_ids: num_sentences x max_sentence_len
        features, _ = self.bert(all_input_ids, attention_mask=all_input_mask,
                                output_hidden_states=False)
        #  features, _ = self.bert(all_input_ids, attention_mask=all_input_mask,
        #                          output_all_encoded_layers=False)
        del _
        # for each word, only keep last encoded token.
        all_end_mask = all_end_mask.to(torch.uint8).unsqueeze(-1)
        features_packed = features.masked_select(all_end_mask)
        features_packed = features_packed.reshape(-1, features.shape[-1])

        assert features_packed.shape[0] == packed_len, "Features: {}, \
            Packed len: {}".format(features_packed.shape[0], packed_len)

        return features_packed

    def annotate_roberta(self, sentences, include_clssep=True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (packed_len, dim).
            packed_len is the total number of words, plus 2 for each sentence
            for [CLS] and [SEP].
        """
        if include_clssep:
            packed_len = sum([(len(s) + 2) for s in sentences])
        else:
            packed_len = sum([len(s) for s in sentences])

        # Each row is the token ids for a sentence, padded with zeros.
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for the last subword for each word.
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        max_sent = 0
        all_avg_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        for s_num, sentence in enumerate(sentences):
            tokens = []
            end_mask = []
            avg_mask = []
            tokens.append("<s>")
            avg_mask.append(0)
            end_mask.append(int(include_clssep))
            current = 1
            for word in sentence:
                word_tokens = self.bert_tokenizer.tokenize(word)
                assert len(word_tokens) > 0, "Unknown word: {} in {}".format(word, sentence)
                for i in range(len(word_tokens)):
                  if i == 0:
                    end_mask.append(1)
                  else:
                    end_mask.append(0)
                if len(word_tokens) > 1:
                  for _ in range(len(word_tokens)):
                    avg_mask.append(current)
                  current = current + 1
                elif len(word_tokens) == 1:
                  avg_mask.append(0)
                #end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("</s>")
            avg_mask.append(0)
            #  print(tokens)
            #  print(avg_mask)
            end_mask.append(int(include_clssep))
            #  print(end_mask)
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(end_mask)] = end_mask
            max_sent = max(max_sent, len(input_ids))
            all_avg_mask[s_num, 0:] = -1
            all_avg_mask[s_num, 0:len(avg_mask)] = avg_mask


        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        all_end_mask = all_end_mask[:, :max_sent]
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask))
        all_avg_mask = all_avg_mask[:, :max_sent]
        #print(all_avg_mask.shape)
        all_avg_mask = all_avg_mask.tolist()
        # all_input_ids: num_sentences x max_sentence_len
        #  features = self.bert(all_input_ids, attention_mask = all_input_mask,
        #                          output_hidden_states = False)['last_hidden_state']
        features = self.bert(all_input_ids, attention_mask = all_input_mask,
                                output_hidden_states = False)[0]

        all_sentences = list()
        for i in range(len(all_avg_mask)):
          subword_sum = np.zeros((768, ))
          divide_by = 0
          for j in range(len(all_avg_mask[i])):
            if all_avg_mask[i][j] == 0:
              f = features[i,j, :]
              f = f.detach().cpu().numpy().tolist()
              all_sentences.append(f)
            elif all_avg_mask[i][j] != 0 and all_avg_mask[i][j] != -1:
              f = features[i, j, :]
              f = f.detach().cpu().numpy()
              subword_sum = np.add(subword_sum, f)
              divide_by = divide_by + 1
              if j <= len(all_avg_mask[i]) - 2:
                if all_avg_mask[i][j+1] != all_avg_mask[i][j] or all_avg_mask[i][j+1] == -1 or  all_avg_mask[i][j+1] == 0:
                  subword_sum = np.divide(subword_sum, divide_by)
                  t = subword_sum.tolist()
                  all_sentences.append(t)
                  subword_sum = np.zeros((768, ))
                  divide_by = 0
              elif j == (len(all_avg_mask[i]) - 1):
                subword_sum = np.divide(subword_sum, divide_by)
                t = subword_sum.tolist()
                all_sentences.append(t)
                subword_sum = np.zeros((768, ))
                divide_by = 0
        all_sentences = torch.tensor(all_sentences, dtype = torch.float32)
        #  print(all_sentences.shape)


        del _
        '''
        # for each word, only keep last encoded token.
        #print(all_end_mask.shape)
        all_end_mask = all_end_mask.to(torch.uint8).unsqueeze(-1)
        #print(all_end_mask.shape)
        features_packed = features.masked_select(all_end_mask)
        #print(features_packed.shape)
        features_packed = features_packed.reshape(-1, features.shape[-1])
        #print(features_packed.shape)

        #assert features_packed.shape[0] == packed_len, "Features: {}, \
        #    Packed len: {}".format(features_packed.shape[0], packed_len)

        return features_packed
        '''
        return all_sentences


def keep_1to1(alignments):
    if len(alignments) == 0:
        return alignments

    counts1 = np.zeros(np.max(alignments[:,0]) + 1)
    counts2 = np.zeros(np.max(alignments[:,1]) + 1)

    for a in alignments:
        counts1[a[0]] += 1
        counts2[a[1]] += 1

    alignments2 = []
    for a in alignments:
        if counts1[a[0]] == 1 and counts2[a[1]] == 1:
            alignments2.append(a)
    return np.array(alignments2)


def load_align_corpus(sent_path, align_path, max_len = 128, max_sent = np.inf):
    sentences_1 = []
    sentences_2 = []
    bad_idx = []
    with open(sent_path) as sent_file:
        """Lines should be of the form
        doch jetzt ist der Held gefallen . ||| but now the hero has fallen .

        Result:
        [
        ['doch', 'jetzt', ...],
        ...
        ]

        [
        ['but', 'now', ...],
        ...
        ]

        If sentences are already in sub-tokenized form, then max_len should be
        512. Otherwise, sentence length might increase after bert tokenization.
        (Bert has a max length of 512.)
        """
        for i, line in enumerate(sent_file):
            if i >= max_sent:
                break

            sent_1 = line[:line.index("|||")].split()
            sent_2 = line[line.index("|||"):].split()[1:]

            if len(sent_1) > max_len or len(sent_2) > max_len:
                bad_idx.append(i)
            else:
                sentences_1.append(sent_1)
                sentences_2.append(sent_2)

    if align_path is None:
        return sentences_1, sentences_2, None

    alignments = []
    with open(align_path) as align_file:
        """Lines should be of the form
        0-0 1-1 2-4 3-2 4-3 5-5 6-6

        Only keeps 1-to-1 alignments.

        Result:
        [
        [[0,0], [1,1], ...],
        ...
        ]
        """
        # need to only keep 1-1 alignments
        for i, line in enumerate(align_file):
            if i >= max_sent:
                break

            if i not in bad_idx:
                alignment = [pair.split('-') for pair in line.split()]
                alignment = np.array(alignment).astype(int)
                alignment = keep_1to1(alignment)

                alignments.append(alignment)
    new_sentences_1 = []
    new_sentences_2 = []
    new_alignments = []
    for a,b,c in zip(sentences_1, sentences_2, alignments):
        d = c.tolist()
        if len(d) < 1:
            continue
        new_sentences_1.append(a)
        new_sentences_2.append(b)
        new_alignments.append(c)
    del sentences_1
    del sentences_2
    del alignments
    return new_sentences_1, new_sentences_2, new_alignments


def partial_sums(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    arr.insert(0, 0)
    return arr[:-1]


def pick_aligned(sent_1, sent_2, align, cls_sep = True):
    """
    sent_1, sent_2 - lists of sentences. each sentence is a list of words.
    align - lists of alignments. each alignment is a list of pairs (i,j).
    """

    idx_1 = partial_sums([len(s) + 2 for s in sent_1])
    idx_2 = partial_sums([len(s) + 2 for s in sent_2])

    align = [a + [i_1, i_2] for a, i_1, i_2 in zip(align, idx_1, idx_2)]
    align = reduce(lambda x, y: np.vstack((x, y)), align)
    align = align + 1 # to account for extra [CLS] at beginning

    if cls_sep:
        # also add cls and sep as alignments
        cls_idx = np.array(list(zip(idx_1, idx_2)))
        sep_idx = (cls_idx - 1)[1:]
        sep_idx_last = np.array([(sum([len(s) + 2 for s in sent_1]) - 1,
                        sum([len(s) + 2 for s in sent_2]) - 1)])
        align = np.vstack((align, cls_idx, sep_idx, sep_idx_last))

    # returns idx_1, idx_2
    # pick out aligned tokens using ann_1[idx_1], ann_2[idx_2]
    return align[:, 0], align[:, 1]


def align_bert_multiple(train, model, model_base,
                        num_sentences, languages, batch_size,
                        model_path,
                        splitbatch_size=4, epochs=1,
                        learning_rate=0.00005, learning_rate_warmup_frac=0.1):
    # Adam hparams from Attention Is All You Need
    trainer = torch.optim.Adam([param for param in model.parameters() if
                                param.requires_grad], lr=1.,
                               betas=(0.9, 0.98), eps=1e-9)

    # set up functions to do linear lr warmup
    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr
    learning_rate_warmup_steps = int(num_sentences * learning_rate_warmup_frac)
    warmup_coeff = learning_rate / learning_rate_warmup_steps
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= learning_rate_warmup_steps:
            print("Warming up, iter {}/{}".format(iteration, learning_rate_warmup_steps))
            set_lr(iteration * warmup_coeff)

    model_base.eval() # freeze and remember initial model

    total_processed = 0
    for epoch in range(epochs):
        for i in range(0, num_sentences, batch_size):
            loss = None
            model.train()
            schedule_lr(total_processed // (len(languages)))
            for j, language in enumerate(languages):
                sent_1, sent_2, align = train[j]
                ii = i % len(sent_1) # cyclic list - datasets may be diff sizes
                ss_1, ss_2 = sent_1[ii:ii+batch_size], sent_2[ii:ii+batch_size]
                aa = align[ii:ii+batch_size]

                # split batch to reduce memory usage
                for k in range(0, len(ss_1), splitbatch_size):
                    s_1 = ss_1[k:k+splitbatch_size]
                    s_2 = ss_2[k:k+splitbatch_size]
                    a = aa[k:k+splitbatch_size]

                    # pick out aligned indices in a packed representation
                    idx_1, idx_2 = pick_aligned(s_1, s_2, a)

                    # compute vectors for each position, pack the sentences
                    # result: packed_len x dim
                    ann_1, ann_2 = model(s_1), model(s_2)
                    ann_2_base = model_base(s_2)
                    #target_shape = ann_1[idx_1].shape[0]
                    #targets = torch.ones(target_shape, 1).to('cuda')
                    loss_1 = F.mse_loss(ann_1[idx_1], ann_2_base[idx_2])
                    loss_2 = F.mse_loss(ann_2, ann_2_base)
                    loss_batch = loss_1 + loss_2
                    if loss is None:
                        loss = loss_batch
                    else:
                        loss += loss_batch
                total_processed += len(ss_1)

            print("Sentences {}-{}/{}, Loss: {}".format(
                    i, min(i+batch_size, num_sentences), num_sentences, loss))
            loss.backward()
            trainer.step()
            trainer.zero_grad()

    torch.save(
            {
                'state_dict': model.state_dict(),
                'trainer': trainer.state_dict(),
            },
            model_path
    )


def normalize(vecs):
    norm = torch.linalg.norm(vecs)
    norm[norm < 1e-5] = 1
    normalized = vecs / norm
    return normalized


def hubness_CSLS(ann_1, ann_2, k=10):
    """
    Computes hubness r(x) of an embedding x, or the mean similarity of x to
    the K closest neighbors in Y. Used for the CSLS metric:
    CSLS(x, y) = 2cos(x,y) - r(x) - r(y)
    which penalizes words with high hubness, or a dense neighborhood.

    Uses k = 10, similarly to https://arxiv.org/pdf/1710.04087.pdf.
    """
    ann_1, ann_2 = normalize(ann_1), normalize(ann_2)
    sim = torch.mm(ann_1, ann_2.T) # words_1 x words_2
    return torch.topk(sim, k, axis=1)[0].mean(axis=1), torch.topk(sim.T, k, axis=1)[0].mean(axis=1)


def bestk_idx_CSLS(x, vecs, vec_hubness, k=5):
    """
    Looks for the k closest vectors using the CSLS metric, which is cosine
    similarity with a hubness penalty.

    Usage:
        hub_1, hub_2 = hubness_CSLS(vecs_1, vecs_2)
        # get word translations for vecs_1[0]
        best_k = bestk_idx_CSLS(vecs_1[0], vecs_2, hub_2)
    """
    x, vecs = normalize(x), normalize(vecs)
    sim = 2 * torch.matmul(vecs, x) - vec_hubness
    return torch.topk(sim, k)[1]


def evaluate_retrieval_context(dev, model):
    with tqdm(total=6) as pbar:
        sent_1, sent_2, align = dev
        idx_1, idx_2 = pick_aligned(sent_1, sent_2, align)
        pbar.update(1)

        model.eval()
        with torch.no_grad():
            ann_1 = model(sent_1)[idx_1]
            ann_2 = model(sent_2)[idx_2]
            pbar.update(1)

            hub_1, hub_2 = hubness_CSLS(ann_1, ann_2)
            pbar.update(1)

            matches_1 = [bestk_idx_CSLS(ann, ann_2, hub_2)[0].detach().cpu().numpy() for ann in ann_1]
            pbar.update(1)

            matches_2 = [bestk_idx_CSLS(ann, ann_1, hub_1)[0].detach().cpu().numpy() for ann in ann_2]
            pbar.update(1)

        acc_1 = np.sum(np.array(matches_1) == np.arange(len(matches_1))) / len(matches_1)
        acc_2 = np.sum(np.array(matches_2) == np.arange(len(matches_2))) / len(matches_2)
        pbar.update(1)

        return acc_1, acc_2


def evaluate_retrieval_noncontext(dev, model):
    with tqdm(total=7) as pbar:
        sent_1, sent_2, align = dev
        idx_1, idx_2 = pick_aligned(sent_1, sent_2, align)
        pbar.update(1)

        all_words = []
        for j in sent_1:
            all_words.extend(["<s>"])
            all_words.extend(j)
            all_words.extend(["</s>"])
        final_1 = []
        for i in idx_1:
            final_1.append(all_words[i])
        all_words = []
        for j in sent_2:
            all_words.extend(["<s>"])
            all_words.extend(j)
            all_words.extend(["</s>"])
        final_2 = []
        for i in idx_2:
            final_2.append(all_words[i])
        found_1 = []
        found_2 = []
        final_idx_1 = []
        final_idx_2 = []
        for i in range(len(final_1)):
            if final_1[i] not in found_1:
                found_1.append(final_1[i])
                found_2.append(final_2[i])
                final_idx_1.append(idx_1[i])
                final_idx_2.append(idx_2[i])
        pbar.update(1)

        model.eval()
        with torch.no_grad():
            ann_1 = model(sent_1)[final_idx_1]
            ann_2 = model(sent_2)[final_idx_2]
            pbar.update(1)

        hub_1, hub_2 = hubness_CSLS(ann_1, ann_2)
        pbar.update(1)

        matches_1 = [bestk_idx_CSLS(ann, ann_2, hub_2)[0].detach().cpu().numpy() for ann in ann_1]
        pbar.update(1)

        matches_2 = [bestk_idx_CSLS(ann, ann_1, hub_1)[0].detach().cpu().numpy() for ann in ann_2]
        pbar.update(1)

        acc_1 = np.sum(np.array(matches_1) == np.arange(len(matches_1))) / len(matches_1)
        acc_2 = np.sum(np.array(matches_2) == np.arange(len(matches_2))) / len(matches_2)
        pbar.update(1)

        return acc_1, acc_2


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='*', default=['es', 'bg', 'fr', 'de', 'el', 'ne'], help='Languages to use')
    parser.add_argument('--mode', choices=['train', 'valid', 'test'], nargs='*', default=['train'], help='Run mode')
    parser.add_argument('--model_type', default='bert-base-multilingual-cased', help='Model type or folder containing a custom one')
    parser.add_argument('--model', default='none', help='Model to save or load')
    args = parser.parse_args()

    #  number of sentences used for experiment train_set+test_set+dev_set for testing they used first 1024 sentences,
    # for dev they used following 1024 sentences and for training they used the following 250000 sentences bur for nepali
    # there are only 67869 sentences.
    # So we should put 67869 here
    num_sent = 3000
    num_dev = 1024
    num_test = 1024

    languages = args.languages
    mode = args.mode
    model_type = args.model_type
    model_path = args.model if args.model.lower() != 'none' else None
    do_lower_case = False

    sent_paths = {
            'es': 'data/europarl-v7.es-en.token.clean',
            'bg': 'data/europarl-v7.bg-en.token.clean',
            'fr': 'data/europarl-v7.fr-en.token.clean',
            'de': 'data/europarl-v7.de-en.token.clean',
            'el': 'data/europarl-v7.el-en.token.clean',
            'ne': 'data/europarl-v7.ne-en.token.clean',
    }

    align_paths = {
            'es': 'data/europarl-v7.es-en.intersect',
            'bg': 'data/europarl-v7.bg-en.intersect',
            'fr': 'data/europarl-v7.fr-en.intersect',
            'de': 'data/europarl-v7.de-en.intersect',
            'el': 'data/europarl-v7.el-en.intersect',
            'ne': 'data/europarl-v7.ne-en.intersect',
    }

    data = [load_align_corpus(sent_paths[lang], align_paths[lang], max_sent=num_sent)
            for lang in tqdm(languages, desc='Loading datasets')]

    if 'train' in mode:
        assert model_path is not None
        #  training codes start here,  comment these following five lines
        #  during validation and testing
        model = WordLevelBert(model_type, do_lower_case, model_path=model_path)
        model_base = WordLevelBert(model_type, do_lower_case, model_path=model_path)

        train = [(sent_1[num_test+num_dev:], sent_2[num_test+num_dev:], align[num_test+num_dev:]) for sent_1, sent_2, align in data]

        #  their default was given 8 please change it i used 4 because of
        #  memory issue
        batch_size = 4
        #  their default was given 1
        epochs = 1
        #  their default was given 0.00005
        learning_rate = 0.00005
        align_bert_multiple(train, model, model_base, num_sent, languages, batch_size, model_path, learning_rate=learning_rate, epochs=epochs)

    if 'valid' in mode:
        #  They didnt use dev set for alignment in their given script so commenting
        #  it out validation codes start here,  comment these following five lines
        #  during training and testing
        dev = [(sent_1[num_test:num_test+num_dev], sent_2[num_test:num_test+num_dev], align[num_test:num_test+num_dev]) for sent_1, sent_2, align in data]
        model = WordLevelBert(model_type, do_lower_case, model_path=model_path)
        for lang, dev_lang in zip(languages, dev):
            print(lang)
            print("Word retrieval accuracy:", evaluate_retrieval_context(dev_lang, model), flush=True)

    if 'test' in mode:
        #  test code start here ,  comment these following five lines during
        #  training and validation
        test = [(sent_1[:num_test], sent_2[:num_test], align[:num_test]) for sent_1, sent_2, align in data]

        model = WordLevelBert(model_type, do_lower_case, model_path=model_path)
        print('\nContext Evaluation: \n')
        for lang, dev_lang in zip(languages, tqdm(test)):
            print("For "+str(lang)+"-en  : ")
            print("Word retrieval accuracy:", evaluate_retrieval_context(dev_lang, model), flush=True)

        print('\nNon Context Evaluation: \n')
        for lang, dev_lang in zip(languages, tqdm(test)):
            print("For "+str(lang)+"-en  : ")
            print("Word retrieval accuracy:", evaluate_retrieval_noncontext(dev_lang, model), flush=True)
