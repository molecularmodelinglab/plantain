def get_activity(batch):
    """ For ease, we also want to support data coming in X, Y tuples """
    if isinstance(batch, tuple):
        return batch[1]
    else:
        return batch.activity