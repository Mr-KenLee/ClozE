

from torch.utils.data import DataLoader, Dataset


class ClozEDataset(Dataset):
    def __init__(self, data, caches):
        super(ClozEDataset, self).__init__()
        self.data = data
        self.caches = caches

    def __getitem__(self, item):

        sample = self.data[item]

        sample_id = sample['sample_id']
        sentence_id = sample['sentence_id']
        factors = sample['factors']

        return {
            'sample_id': sample_id,
            'sentence_id': sentence_id,
            'document': self.caches[sample_id]['document'],
            'summary': self.caches[sample_id]['summary'][sentence_id]['input_ids'],
            'offset_mapping': self.caches[sample_id]['summary'][sentence_id]['offset_mapping'],
            'factors': factors
        }

    def __len__(self):
        return len(self.data)


def get_eval_loader(processed_data, caches, batch_size):
    return DataLoader(ClozEDataset(processed_data, caches), batch_size=batch_size, shuffle=False, collate_fn=lambda x:x)