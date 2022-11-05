

from metric import ClozEMetric

scorer = ClozEMetric(cloze_model_path='ClozE-roberta-base-cnndm',
                     fact_extractor='en_core_web_trf',
                     use_gpu=True)


documents = ['A diet rich in oily fish , whole grains , lean protein , fruit and vegetables should provide enough nutrients .']
summaries = ['A diet rich in oily fish should provide enough nutrients .']


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


