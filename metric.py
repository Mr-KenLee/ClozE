import torch
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from dataset import get_eval_loader
from extractor import FactualFactorExtractor
from scorer import ClozEScorer
from utils import get_tokenization_caches, \
    parse_eval_batch, mask_factors_with_mapping, create_cloze_model_input_tensors


class ClozEMetric:
    def __init__(self,
                 cloze_model_path='ClozE-roberta-base-cnndm',
                 fact_extractor='en_core_web_trf',
                 use_gpu=True,
                 ):
        """
        :param cloze_model_path: the path of cloze model in ClozE
        :param fact_extractor: the model name that Spacy will load in
        :param use_gpu: whether use GPU to compute
        """

        # hyper-parameters
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'

        # load cloze model and tokenizer
        print(f'Load cloze model from {cloze_model_path}.')
        self.model = RobertaForMaskedLM.from_pretrained(cloze_model_path).to(self.device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(cloze_model_path)

        # load factual factor extractor
        print(f'Load {fact_extractor} factual factor extractor.')
        self.extractor = FactualFactorExtractor(fact_extractor, use_gpu)

        # load scorer
        self.scorer = ClozEScorer()

        self.model.eval()
        self.max_len = self.model.config.max_position_embeddings - 2

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def normalize_inputs(self, batch_document, batch_summary, batch_offset_mapping, batch_factors):

        max_len = self.max_len - 4  # <s>...</s></s>...</s>
        batch_concatenation = []
        batch_inputs = []
        batch_positions = []

        # concatenate document and
        for document, summary, offset_mapping, factors in zip(batch_document,
                                                              batch_summary,
                                                              batch_offset_mapping,
                                                              batch_factors):
            # truncation: longest first
            document_len = len(document)
            summary_len = len(summary)

            while document_len + summary_len > max_len:
                if document_len >= summary_len:
                    document_len -= 1
                else:
                    summary_len -= 1

            concatenation = [self.cls_token_id] + document[:document_len] + 2*[self.sep_token_id] + summary[:summary_len] + [self.sep_token_id]
            inputs, positions = mask_factors_with_mapping(concatenation, offset_mapping, factors, document_len+3, self.mask_token_id)

            batch_inputs.append(inputs)
            batch_positions.append(positions)
            batch_concatenation.append(concatenation)

        input_ids, attention_mask, labels = create_cloze_model_input_tensors(batch_inputs, batch_concatenation, self.pad_token_id)

        return input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), batch_positions

    def score(self,
              documents,
              summaries,
              k=1,
              selection='entity_first',
              granularity='sentence',
              criterion='f1_score',
              use_confidence=True,
              use_sentencizer=False,
              summary_strategy='average',
              sentence_strategy='average',
              confidence_strategy='average',
              alpha=0.5,
              beta=0.5,
              eval_batch_size=8,
              verbose=True,
              use_tqdm=True):

        processed_data = self.extractor.extract(documents, summaries, k, granularity, selection, use_tqdm)

        caches, text_caches = get_tokenization_caches(processed_data, self.tokenizer, max_len=self.max_len, use_tqdm=use_tqdm)
        eval_loader = get_eval_loader(processed_data, caches, batch_size=eval_batch_size)

        if use_tqdm:
            bar = tqdm(range(len(eval_loader)), desc=f'Evaluating {len(eval_loader.dataset)} samples', ncols=150)
        else:
            print(f'Evaluating {len(eval_loader.dataset)} samples with {len(eval_loader)} steps...')
            bar = range(len(eval_loader))

        eval_results_dict = {}
        for _, eval_batch in zip(bar, eval_loader):
            batch_sample_id, batch_sentence_id, batch_document, batch_summary, batch_offset_mapping, batch_factors \
                = parse_eval_batch(eval_batch)

            input_ids, attention_mask, labels, positions = self.normalize_inputs(batch_document, batch_summary, batch_offset_mapping,
                                                                                 batch_factors)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            distributions = torch.softmax(outputs.logits, dim=-1)
            max_probabilities = torch.max(distributions, dim=-1)

            probabilities = max_probabilities.values
            predicts = max_probabilities.indices

            # process each sample
            for batch_idx in range(len(predicts)):

                sample_id = batch_sample_id[batch_idx]
                sentence_id = batch_sentence_id[batch_idx]

                if sample_id not in eval_results_dict:
                    eval_results_dict[sample_id] = {
                        'document': text_caches[sample_id]['document'],
                        'summary': {}
                    }

                if sentence_id not in eval_results_dict[sample_id]['summary']:
                    eval_results_dict[sample_id]['summary'][sentence_id] = {
                        'sentence': text_caches[sample_id]['summary'][sentence_id],
                        'factors': [],
                        'labels': [],
                        'predicts': [],
                        'probabilities': []
                    }

                for position_idx in range(len(positions[batch_idx])):
                    start, end = positions[batch_idx][position_idx]

                    # Because of the tiny difference between original text and the text processed by SpaCy,
                    # we save both of them to display errors and compute more accurate scores.
                    eval_results_dict[sample_id]['summary'][sentence_id]['factors'].append(
                        batch_factors[batch_idx][position_idx])
                    eval_results_dict[sample_id]['summary'][sentence_id]['labels'].append(
                        labels[batch_idx, start:end].tolist())
                    eval_results_dict[sample_id]['summary'][sentence_id]['predicts'].append(
                        predicts[batch_idx, start:end].tolist())
                    eval_results_dict[sample_id]['summary'][sentence_id]['probabilities'].append(
                        probabilities[batch_idx, start:end].tolist())

        final_scores = self.scorer.evaluate(eval_results_dict,
                                            self.tokenizer,
                                            criterion,
                                            use_confidence,
                                            use_sentencizer,
                                            summary_strategy,
                                            sentence_strategy,
                                            confidence_strategy,
                                            alpha, beta, verbose)

        if verbose:
            return final_scores
        else:
            return [f['score'] for f in final_scores]

    @staticmethod
    def display_results(results):
        for i, result in enumerate(results):
            print(f'####################Sample {i}###############################')
            print('ClozE Score:', result['score'])
            print('Document:', result['infos']['document'])
            for info in result['infos']['summary']:
                print('Sentence:', info['sentence'])
                print('Sentence Score:', info['score'])
                print('-----------------------------------------------')
                for comp in info['comparision']:
                    for key in comp:
                        print(f"\t{key}: {comp[key]}")
                    print('-----------------------------------------------')
            print('########################################################')
