# Copyright 2020 University of Toronto, all rights reserved

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''

    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.
    # with torch.no_grad():
    # 1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
    #    keeping track of what id the loss considers "padding"
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.target_eos)
    total_loss = 0

    # 2. For every iteration of the `dataloader` (which yields triples
    #    ``F, F_lens, E``)
    for F, F_lens, E in dataloader:
        # 1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
        #       for ``F_lens`` and ``E``.
        F = F.to(device)
        F_lens = F_lens.to(device)
        E = E.to(device)

        # 2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
        optimizer.zero_grad()

        # 3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
        #     probabilities.
        logits = model(F, F_lens, E, 'ignore')
        #TODO: print("logits: {},".format(logits.shape)) should return (16, 100, 20000) instead of (100, 320000)

        # 4. Modifies ``E`` for the loss function, getting rid of a token and
        #     replacing excess end-of-sequence tokens with padding using
        # ``model.get_target_padding_mask()`` and ``torch.masked_fill``
        pad_mask = model.get_target_padding_mask(E)
        E = E.masked_fill(pad_mask, model.target_eos)

        # 5. Flattens out the sequence dimension into the batch dimension of both
        #     ``logits`` and ``E``
        logits_flat = logits.view(-1, logits.size(-1)) # (T - 1, N, V) -> ((T-1)*N, V)
        # print("T: {}, N: {}, V: {}".format(E.shape[0], E.shape[1], model.target_vocab_size))
        E = E.transpose(0, 1)[:, 1:].reshape(-1)
        # print("E: {}, logits_flat: {}".format(E.shape, logits_flat.shape))

        # 6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
        loss = loss_fn(logits_flat, E)
        total_loss += loss

        # 7. Calls ``loss.backward()`` to backpropagate gradients through
        #     ``model``
        loss.backward()

        # 8. Calls ``optim.step()`` to update model parameters
        optimizer.step()

        # To prevent CUDA memory errors part way through training
        del F, F_lens, E, logits, loss

    # 3. Returns the average loss over sequences
    avg_bleu = total_loss / len(dataloader.data)
    return avg_bleu


def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, N)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', N)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    score, ngram, N = 0, 4, E_ref.shape[1]
    refs, cands = E_ref.tolist(), E_cand.tolist()

    for ref, cand in zip(refs, cands):
        ref = ref.squeeze(0)
        ref = ref[((ref != target_sos) & (ref != target_eos)
                   ).nonzero().squeeze()].tolist()
        if not isinstance(ref, list):
            ref = [ref]

        cand = cand.squeeze(0)
        cand = cand[((cand != target_sos) & (cand != target_eos)
                     ).nonzero().squeeze()].tolist()
        if not isinstance(cand, list):
            cand = [cand]

        score += a2_bleu_score.BLEU_score(ref, cand, ngram)
    return score


def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''

    with torch.no_grad():
        total_score, total_batches = 0, 0

        # For every iteration of the `dataloader` (which yields triples``F, F_lens, E_ref``):
        for F, F_lens, E_ref in tqdm(dataloader):
            # 1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
            # for ``F_lens``. No need for ``E_cand``, since it will always be
            # compared on the CPU.
            batch_size = F.shape[1]
            F, F_lens = F.to(device), F_lens.to(device)

            # 2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
            # shape (T', N, self.beam_width)
            b_1 = model(F, F_lens, on_max='halt')

            # 3. Extracts the top path per beam as ``E_cand = b_1[..., 0]`` of shape (T', N)
            E_cand = b_1[:, :, 0]

            # 4. Computes the total BLEU score of the batch using func: `compute_batch_total_bleu`
            total_score += compute_batch_total_bleu(
                E_ref, E_cand, target_sos, target_eos)
            total_batches += 1

            del F, F_lens, E_cand, b_1, E_ref

    # Returns the average per-sequence BLEU score
    avg_bleu = total_score / \
        len(dataloader.data) if len(dataloader.data) != 0 else 0

    return avg_bleu
