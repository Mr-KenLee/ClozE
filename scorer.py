
import torch
import torch.nn

from transformers import AutoTokenizer
from utils import f1_score, em_score, soft_em_score


class ClozEScorer:
    def __init__(self):
        print('Construct ClozScorer successfully.')

    def compute_factor_score(self, label, predict, tokenizer, criterion):

        # Uniform letter case
        n_label_str = tokenizer.decode(label).strip()
        n_label = tokenizer.encode(' ' + n_label_str.lower(), add_special_tokens=False)
        n_predict_str = tokenizer.decode(predict).strip()
        n_predict = tokenizer.encode(' ' + n_predict_str.lower(), add_special_tokens=False)

        if not n_label_str and not n_predict_str:
            return 1.0, n_predict_str, n_label_str

        if criterion == 'f1_score':
            return f1_score(n_label, n_predict), n_predict_str, n_label_str
        elif criterion == 'em_score':
            return em_score(n_label, n_predict), n_predict_str, n_label_str
        elif criterion == 'soft_em_score':
            return soft_em_score(n_label, n_predict), n_predict_str, n_label_str
        else:
            return f1_score(n_label, n_predict), n_predict_str, n_label_str

    def compute_score(self, scores, strategy):

        if not scores:
            return 1.0

        if strategy == 'average':
            return sum(scores) / len(scores)
        elif strategy == 'min':
            return min(scores)
        elif strategy == 'max':
            return max(scores)
        else:
            return sum(scores) / len(scores)

    def evaluate(self,
                 eval_results_dict,
                 tokenizer,
                 criterion,
                 use_confidence,
                 use_sentencizer,
                 summary_strategy,
                 sentence_strategy,
                 confidence_strategy,
                 alpha,
                 beta,
                 verbose):

        results = []

        # To prevent the order from being messed up, here we use range to iter.
        for sample_idx in range(len(eval_results_dict)):

            sample = eval_results_dict[sample_idx]
            document = sample['document']
            summary = []

            summary_infos = sample['summary']

            result = {}
            sentence_scores = []

            for sentence_idx in range(len(summary_infos)):

                infos = {'sentence': summary_infos[sentence_idx]['sentence'],
                         'comparision': [],
                         'score': 1.0}

                summary_info = summary_infos[sentence_idx]
                factor_scores = []

                for factor, label, predict, probability in zip(summary_info['factors'],
                                                               summary_info['labels'],
                                                               summary_info['predicts'],
                                                               summary_info['probabilities']):

                    score, answer, origin = self.compute_factor_score(label, predict, tokenizer, criterion)

                    if use_confidence:
                        confidence = self.compute_score(probability, confidence_strategy)
                        if score < beta and confidence < alpha:
                            score = 0

                    infos['comparision'].append({
                        'factor': factor['text'],
                        'answer': answer,
                        'score': score,
                    })

                    factor_scores.append(score)

                # sentence_score = self.compute_score(factor_scores, sentence_strategy)

                sentence_scores.append(factor_scores)

                infos['score'] = self.compute_score(factor_scores, sentence_strategy)
                summary.append(infos)

            if use_sentencizer:
                sentence_scores = [infos['score'] for infos in summary]
            else:
                sentence_scores = sum(sentence_scores, [])

            result['score'] = self.compute_score(sentence_scores, summary_strategy)

            if verbose:
                result['infos'] = {
                    'document': document,
                    'summary': summary,
                }

            results.append(result)

        return results
