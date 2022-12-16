import importlib

import spacy
from tqdm import tqdm
from utils import convert_word_idx_to_char_idx

class FactualFactorExtractor:
    def __init__(self, fact_extractor='en_core_web_trf', use_gpu=False, device=0):

        if use_gpu:
            spacy.require_gpu(device)

        # load nlp tool
        # it can be loaded by "import en_core_web_trf" or "spacy.load('en_core_web_trf')"
        try:
            module = importlib.import_module(fact_extractor)
            self.nlp = module.load()
        except ImportError:
            self.nlp = spacy.load(fact_extractor)

    def extract_factual_factors(self,
                                documents,
                                summaries,
                                granularity='sentence',
                                use_tqdm=True):
        """
        1. Split sentences by SpaCy in order to match the index in entities or nouns extracted by it.
        2. Extract factual factors from summary by SpaCy.
        """

        processed_data = []

        if use_tqdm:
            bar = tqdm(list(zip(documents, summaries)), desc=f'Extracting factual factors', ncols=150)
        else:
            print(f'Extracting factual factors from {len(documents)} samples...')
            bar = list(zip(documents, summaries))

        # process each pair of document and summary
        for i, (r_document, r_summary) in enumerate(bar):

            document = ' '.join(list(filter(lambda x: x, r_document.split(' '))))
            summary = ' '.join(list(filter(lambda x: x, r_summary.split(' '))))

            seg_document = [word.text for word in self.nlp(document)]

            bias = 0  # sentence bias
            summary_sentences = []
            summary_entities = []
            summary_nouns = []

            if granularity == 'sentence':
                seg_sentences = self.nlp(summary).sents
            else:
                seg_sentences = [self.nlp(summary)]

            for seg_sentence in seg_sentences:
                seg_summary_sentence = [word.text for word in seg_sentence]
                entities = [{'text': ent.text,
                             'start': ent.start - bias,
                             'end': ent.end - bias} for ent in seg_sentence.ents]
                nouns = [{'text': noun_chunk.text,
                          'start': noun_chunk.start - bias,
                          'end': noun_chunk.end - bias} for noun_chunk in
                         seg_sentence.noun_chunks]
                bias += len(seg_summary_sentence)

                # convert the word indices to char indices
                # 'I am a good man.' (good, 3, 4) to (good, 7, 11) 
                entities = convert_word_idx_to_char_idx(seg_summary_sentence, entities)
                nouns = convert_word_idx_to_char_idx(seg_summary_sentence, nouns)

                summary_sentences.append(' '.join(seg_summary_sentence))    # merge to a string
                summary_entities.append(entities)
                summary_nouns.append(nouns)

            processed_data.append({
                'document': ' '.join(seg_document),
                'summary_sentences': summary_sentences,
                'summary_entities': summary_entities,
                'summary_nouns': summary_nouns
            })

        return processed_data

    def mixture_factual_factors(self, factors1, factors2):
        """
        Drop out the factors2's factors which overlap factors1's factors.
        """
        candidate_factors = []

        for j in range(len(factors2)):
            for i in range(len(factors1)):
                if i == 0:
                    if factors2[j]['end'] <= factors1[i]['start']:
                        candidate_factors.append(factors2[j])
                        break
                else:
                    if factors2[j]['start'] >= factors1[i - 1]['end'] and factors2[j]['end'] <= factors1[i]['start']:
                        candidate_factors.append(factors2[j])
                        break

            if factors1 and factors2[j]['start'] >= factors1[-1]['end']:
                candidate_factors.extend(factors2[j:])
                break

        if not factors1:
            factors = factors2
        else:
            factors = factors1 + candidate_factors
            factors = list(sorted(factors, key=lambda x: x['start']))

        return factors

    def select_factual_factors(self, entities, nouns, selection):

        if selection == 'entity':
            return entities
        elif selection == 'noun':
            return nouns
        elif selection == 'noun_first':
            return self.mixture_factual_factors(nouns, entities)
        else:
            return self.mixture_factual_factors(entities, nouns)

    def block(self, factors, k):
        blocks = []
        block = []
        for factor in factors:
            block.append(factor)
            if len(block) == k:
                blocks.append(block)
                block = []

        if block:
            blocks.append(block)

        return blocks

    def block_factual_factors(self, data, k, selection, use_tqdm):

        processed_data = []

        if use_tqdm:
            bar = tqdm(data, desc=f'Blocking factual factors by k={k}', ncols=150)
        else:
            print(f'Blocking factual factors by k={k} from {len(data)} samples...')
            bar = data

        for i, sample in enumerate(bar):
            for j, (sentence, entities, nouns) in enumerate(
                    zip(sample['summary_sentences'], sample['summary_entities'], sample['summary_nouns'])):
                factors = self.select_factual_factors(entities, nouns, selection)
                factors = self.block(factors, k)
                factors = factors if factors else [[]]

                for sub_factors in factors:
                    processed_data.append({
                        'sample_id': i,
                        'sentence_id': j,
                        'document': sample['document'],
                        'summary': sentence,
                        'factors': sub_factors
                    })

        return processed_data

    def extract(self,
                documents,
                summaries,
                k,
                granularity,
                selection,
                use_tqdm):
        """
        1. Extract factual factors from summary.
        2. Block the samples by k.
        """
        processed_data = self.extract_factual_factors(documents, summaries, granularity, use_tqdm)
        processed_data = self.block_factual_factors(processed_data, k, selection, use_tqdm)

        return processed_data
