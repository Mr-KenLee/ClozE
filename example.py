
import json

from metric import ClozEMetric
from scipy.stats import pearsonr

scorer = ClozEMetric(cloze_model_path='ClozE-roberta-base-cnndm',
                     fact_extractor='en_core_web_trf',
                     use_gpu=True)

with open('./test_data_qags_cnndm.json', 'r', encoding='utf8') as file:
    data = json.load(file)

n = 10

documents = [d['source'] for d in data]
summaries = [d['summary'] for d in data]

annotation = [d['score'] for d in data]

predicts = scorer.score(documents,
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
                        use_tqdm=True)

scorer.display_results(predicts)
scores = [r['score'] for r in predicts]

print(pearsonr(annotation, scores))
