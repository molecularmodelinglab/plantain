def get_activity(batch):
    if isinstance(batch, tuple):
        return batch[1]
    else:
        return batch.activity