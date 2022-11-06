import torch
import torch.nn as nn
import copy
import sklearn.metrics as metrics

from transformers import RobertaTokenizerFast
from tqdm import tqdm

def get_tokenization_caches(data, tokenizer, max_len=512, use_tqdm=True):
    """
    The format of a sample in data is:
        {
            'sample_id': 0,
            'sentence_id': 0,
            'document': 'This is a document.',
            'summary': 'This is a sentence in a summary.',
            'factors': [('factor_0', 0, 3)]

        }

    To reduce the number of repeated tokenize documents and sub-sentences,
    we will use this function to generate caches instead of function 'batch_encode_plus'.
    """
    def tokenize_text(text):
        outputs = tokenizer.encode_plus(
            text=text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True
        )

        return outputs['input_ids'], outputs['offset_mapping']

    caches = {}
    text_caches = {}

    if use_tqdm:
        bar = tqdm(data, desc=f'Tokenizing {len(data)} samples', ncols=150)
    else:
        print(f'Tokenizing {len(data)} samples...')
        bar = data

    for sample in bar:
        sample_id, sentence_id = sample['sample_id'], sample['sentence_id']

        if sample_id not in caches:
            caches[sample_id] = {
                'document': tokenize_text(sample['document'])[0],
                'summary': {}
            }

            text_caches[sample_id] = {
                'document': sample['document'],
                'summary': {}
            }

        if sentence_id not in caches[sample_id]['summary']:
            input_ids, offset_mapping = tokenize_text(sample['summary'])
            caches[sample_id]['summary'][sentence_id] = {
                'input_ids': input_ids,
                'offset_mapping': offset_mapping
            }
            text_caches[sample_id]['summary'][sentence_id] = sample['summary']

    return caches, text_caches


def parse_eval_batch(eval_batch):
    """
    {
        'sample_id': 0,
        'sentence_id': 0,
        'document': [4,5,6,7,8,9],
        'summary': [4,5,6,7,8],
        'offset_mapping': [[0,1],[1,5],[6,7],[7,8],[8,15]],
        'factors': [('factor_0', 1, 7)]
    }
    """

    batch_sample_id = [eval_sample['sample_id'] for eval_sample in eval_batch]
    batch_sentence_id = [eval_sample['sentence_id'] for eval_sample in eval_batch]
    batch_document = [eval_sample['document'] for eval_sample in eval_batch]
    batch_summary = [eval_sample['summary'] for eval_sample in eval_batch]
    batch_offset_mapping = [eval_sample['offset_mapping'] for eval_sample in eval_batch]
    batch_factors = [eval_sample['factors'] for eval_sample in eval_batch]

    return batch_sample_id, batch_sentence_id, batch_document, batch_summary, batch_offset_mapping, batch_factors


def get_head_position(offset_mapping):
    """
    There will be four [0,0] positions for <s> and </s> which are in "<s> s11,s12,... </s></s> s21,s22,...</s>".
    We need to find the 2nd [0,0] position, which is the beginning of the second sentence.
    """
    flag = True
    for i in range(0, len(offset_mapping)):
        if offset_mapping[i, 0] == offset_mapping[i, 1] == 0:
            if flag:
                flag = False
            else:
                return i+1
    return 0


def get_masked_span_scope(factor, offset_mapping, head):

    start = -1
    end = -1

    for i in range(head, len(offset_mapping)):
        if offset_mapping[i][0] == offset_mapping[i][1] == 0:
            continue
        if offset_mapping[i][0] == factor['start']:
            start = i
        if offset_mapping[i][1] == factor['end']:
            end = i + 1
        if start != -1 and end != -1:
            break

    return start, end


def mask_factors_with_mapping(labels, offset_mapping, factors, head, mask_token_id):

    inputs = copy.deepcopy(labels)
    positions = []

    summary_start = 0
    for factor in factors:
        start, end = get_masked_span_scope(factor, offset_mapping, summary_start)
        summary_start = end

        for k in range(start+head, end+head):
            inputs[k] = mask_token_id

        positions.append([head+start, head+end])

    return inputs, positions


def create_cloze_model_input_tensors(batch_inputs, batch_concatenation, pad_token_id):

    lengths = [len(b) for b in batch_inputs]
    max_len = max(lengths)

    input_ids = torch.zeros((len(batch_inputs), max_len), dtype=torch.long) + pad_token_id
    attention_mask = torch.zeros((len(batch_inputs), max_len))
    labels = torch.zeros((len(batch_inputs), max_len), dtype=torch.long) + pad_token_id

    for i in range(len(batch_inputs)):
        input_ids[i, :lengths[i]] = torch.LongTensor(batch_inputs[i])
        attention_mask[i, :lengths[i]] = 1
        labels[i, :lengths[i]] = torch.LongTensor(batch_concatenation[i])

    return input_ids, attention_mask, labels


def em_score(src_factor_tokens, pred_factor_tokens):
    return src_factor_tokens == pred_factor_tokens


def soft_em_score(src_factor_tokens, pred_factor_tokens):
    return metrics.accuracy_score(src_factor_tokens, pred_factor_tokens)


def f1_score(src_factor_tokens, pred_factor_tokens):
    recall = 0
    precision = 0

    for i in pred_factor_tokens:
        if i in src_factor_tokens:
            recall += 1

    for i in src_factor_tokens:
        if i in pred_factor_tokens:
            precision += 1

    recall /= len(pred_factor_tokens)
    precision /= len(src_factor_tokens)

    return 2 * recall * precision / (recall + precision + 1e-12)


def convert_word_idx_to_char_idx(seg_summary, factors):

    n_factors = []

    for factor in factors:

        bias = int(factor['start'] > 0)
        start = len(' '.join(seg_summary[:factor['start']])) + bias
        end = len((' '.join(seg_summary[:factor['end']])))

        n_factors.append({
            'text': factor['text'],
            'start': start,
            'end': end
        })

    return n_factors
