import torch


def sequence_mask(lengths, max_len=None):
    """Same as tf.sequence_mask, Returns a mask tensor representing the first N positions of each cell.

    Parameters
    ----------
    lengths : _type_
        integer tensor, all its values <= maxlen.
    max_len : _type_, optional
        scalar integer tensor, size of last dimension of returned tensor. Default is the maximum value in lengths.

    Returns
    -------
    _type_
        A mask tensor of shape lengths.shape + (maxlen,)
    
    Examples:
    ---------
    >>> sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                #  [True, True, True, False, False],
                                #  [True, True, False, False, False]]

    >>> sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                    #   [True, True, True]],
                                    #  [[True, True, False],
                                    #   [False, False, False]]]
    """

    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)


def gather_nd(params, indices):
    """_summary_

    Parameters
    ----------
    params : _type_
        _description_
    indices : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    
    Examples:
    ---------
    >>> gather_nd(
    ...           params = [['a', 'b', 'c'],
    ...                     ['d', 'e', 'f']]).numpy(),
    ...           indices = [[1],
    ...                      [0]])
    """
    newshape = indices.shape[:-1] + params.shape[indices.shape[-1]:]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([params.__getitem__(tuple(i)) for i in indices])
    return out.reshape(newshape)
