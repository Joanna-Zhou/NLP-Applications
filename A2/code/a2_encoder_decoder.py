# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        '''Initialize the parameterized submodules of this network'''
        # initialize parameterized submodules here: rnn, embedding
        # using:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        #   torch.nn.{LSTM, GRU, RNN, Embedding}

        # embedding : torch.nn.Embedding
        #     A layer that extracts learned token embeddings for each index in a token sequence.
        #     It must not learn an embedding for padded tokens.
        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        # rnn : {torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM}
        #     A layer corresponding to the recurrent neural network that
        #     processes source word embeddings. It must be bidirectional.
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(self.word_embedding_size,
                                     self.hidden_state_size,
                                     self.num_hidden_layers,
                                     dropout=self.dropout,
                                     bidirectional=True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(self.word_embedding_size,
                                    self.hidden_state_size,
                                    self.num_hidden_layers,
                                    dropout=self.dropout,
                                    bidirectional=True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(self.word_embedding_size,
                                    self.hidden_state_size,
                                    self.num_hidden_layers,
                                    dropout=self.dropout,
                                    bidirectional=True)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        x = self.embedding(F)

        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        print(
            "-----\nIn get_all_rnn_inputs, F: {} -> x: {}\n-----".format(F.shape, x.shape))
        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Pack hidden states
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens)
        # compute all final hidden states for provided input sequence.
        outputs, hidden = self.rnn(x_packed)
        # Unpack hidden states
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # TODO: check if we need to manually concatenate
        # TODO: if we do, use https://bastings.github.io/annotated_encoder_decoder/
        # print("-----\nIn get_all_hidden_states, hidden before concat: {}\n-----".format(hidden.shape))
        # # Concatenate both forward and backward states
        # h_1, h_2 = hidden[0], hidden[1]
        # hidden = torch.cat((h_1, h_2), dim=1)

        # x is of shape (S, N, I)
        # h (output) is of shape (S, N, 2 * H)
        print("-----\nIn get_all_hidden_states, x: {} -> h: {}\n-----".format(x.shape, hidden.shape))

        # Sum the both forward and backward
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return hidden


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # embedding : torch.nn.Embedding
        #     A layer that extracts learned token embeddings for each index in
        #     a token sequence. It must not learn an embedding for padded tokens.
        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        # cell : {torch.nn.RNNCell, torch.nn.GRUCell, torch.nn.LSTMCell}
        #     A layer corresponding to the recurrent neural network that
        #     processes target word embeddings into hidden states. We only define
        #     one cell and one layer
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size + 2 * self.hidden_state_size,
                                          2 * self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size + 2 * self.hidden_state_size,
                                         2 * self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size + 2 * self.hidden_state_size,
                                         2 * self.hidden_state_size)

        # ff : torch.nn.Linear
        #     A fully-connected layer that converts the decoder hidden state
        #     into an un-normalized log probability distribution over target
        #     words
        self.ff = torch.nn.Linear(in_features=self.hidden_state_size,
                                  out_features=self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        '''Get the initial decoder hidden state, prior to the first input'''
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each direction:
        hidden_state_mid = self.hidden_state_size // 2
        # h_unpadded = h[F_lens]  # (N, 2*H)
        # print("-----\nIn get_first_hidden_state, h_unpadded: {}".format(h_unpadded.shape))

        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # Take each hidden state at F_len[n-1]
        forward = h_unpadded[F_lens-1, :, :hidden_state_mid]

        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        backward = h[0, :, hidden_state_mid:]

        # TODO: check concatination dimension
        # TODO: check what i need to do with LSTM
        htilde_tm1 = torch.cat((forward, backward), dim=1)
        # h is of shape(S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        print("-----\nIn get_first_hidden_state, h: {} -> htilde_tm1: {}".format(h.shape, htilde_tm1.shape))
        print("-- vs. expected dimension: h: (S, N, {}) -> htilde_tm1: (N, {})\n-----".format(
            self.hidden_state_size * 2, self.hidden_state_size * 2))

        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # xtilde_t : torch.FloatTensor
        #     A float tensor of shape ``(N, Itilde)`` denoting the current input
        #     to the decoder RNN.
        # ``xtilde_t[n, :self.word_embedding_size]``:
        #       A word embedding for ``E_tm1[n]``.
        #       If ``E_tm1[n] == self.pad_id``, then ``xtilde_t[n] == 0.``.
        # TODO: make sure that embedding does masked the padded ones
        embedded = self.embedding(E_tm1)
        xtilde_t = torch.stack((embedded, htilde_tm1))

        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        print("-----In get_current_rnn_input, htilde_tm1: {} -> xtilde_t: {}".format(
            htilde_tm1.shape, xtilde_t.shape))
        print("-- vs. expected dimension: htilde_tm1: (S, N, {}) -> xtilde_t: (N, {})\n-----".format(
            self.hidden_state_size * 2, self.word_embedding_size + self.hidden_state_size))

        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        '''Calculate the decoder's current hidden state

        Converts `E_tm1` to embeddings, and feeds those embeddings into
        the recurrent cell alongside `htilde_tm1`.'''
        # htilde_tm1 : torch.FloatTensor or tuple
        #     If this decoder doesn't use an LSTM cell, `htilde_tm1` is a float
        #     tensor of shape ``(N, self.hidden_state_size)``, where
        #     ``htilde_tm1[n]`` corresponds to ``n``-th element in the batch.
        #     If this decoder does use an LSTM cell, `htilde_tm1` is a pair of
        #     float tensors corresponding to the previous hidden state and the
        #     previous cell state.

        htilde_t = self.decoder(xtilde_t, htilde_tm1)

        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        print("-----\nIn get_current_hidden_state, htilde_tm1: {} -> xtilde_t: {}\n-----".format(
            htilde_tm1.shape, htilde_t.shape))

        return htilde_t

    def get_current_logits(self, htilde_t):
        '''Calculate an un-normalized log distribution over target words

        Parameters
        ----------
        htilde_t : torch.FloatTensor
            A float tensor of shape ``(N, self.hidden_state_size)`` of the
            decoder's current hidden state (excludes the cell state in the
            case of an LSTM).

        Returns
        -------
        logits_t : torch.FloatTensor
            A float tensor of shape ``(N, self.target_vocab_size)``.
            ``logits_t[n]`` is an un-normalized distribution over the next
            target word for the ``n``-th sequence:
            ``Pr_b(i) = softmax(logits_t[n])``
        '''
        # determine un-normalized log-probability distribution over output tokens for current time step.
        logits = self.ff(htilde_t)

        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        print("-----\nIn get_current_logits, htilde_tm1: {} -> logits: {}\n-----".format(
            htilde_t.shape, logits.shape))
        return logits


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # * Embedding
        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        # * RNN
        input_size = self.word_embedding_size + 2 * self.hidden_state_size
        hidden_size = enc_output_size = 2 * self.hidden_state_size

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size, hidden_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size, hidden_size)

        # * Attention
        self.attn = torch.nn.Linear(enc_output_size, hidden_size)
        self.energy = nn.Linear(hidden_size, 1)

        # * Logit
        self.ff = torch.nn.Linear(in_features=hidden_size, out_features=self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        # h : (S, N, self.hidden_state_size) -> htilde_0 : (N, self.hidden_state_size)
        # since all is initialized to zero, which h to take doesn't matter
        return torch.zeros_like(h[-1], device=h.device)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        c_t = self.attend(htilde_t, h, F_lens)
        xtilde_t = torch.stack((embedded, c_t))
        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        alphas = self.get_attention_weights(htilde_t, h, F_lens)
        c_t = torch.bmm(alphas, h)
        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energytopk_logpy_t()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energytopk_logpy_t(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energytopk_logpy_t(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        S, N = h.shape[0], h.shape[1]
        energy = torch.zeros(S, N)
        for s in range(S):
            energy[s] = torch.nn.functional.cosine_similarity(htilde_t, h[s], dim=1)
        return energy


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_self.hidden_state_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)

        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size,
                                     cell_type=self.cell_type)  # TODO: Check if this should be the same as encoder hidden size

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the sequence.
        # h is of shape (S, N, 2 * H)
        # logits (output) is of shape (T - 1, N, Vo)

        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)

        # F_lens is of shape (N,)
        batch_size = F_lens.shape[0]

        # F_lens is of shape (N,)
        T, N = E.shape[0], E.shape[1]
        # TODO: Thank about how/if we need to pad
        # A float tensor of shape ``(S, N, 2 * self.encoder_hidden_size)`` of
        # hidden states of the encoder. ``h[s, n, i]`` is the
        # ``i``-th index of the encoder RNN's last hidden state at time ``s``
        # of the ``n``-th sequence in the batch. The states of the
        # encoder have been right-padded such that ``h[F_lens[n]:, n]``
        # should all be ignored.

        # Initilize hidden states
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))

        # Initialize output results (i.e., just logits here, no token needed)
        logits_list = []

        for t in range(T-1):
            # The following "forward" method is modified from the decoder's forward()
            # ! This is the teacher enforcing part, where E_tm1 <- E
            E_tm1 = E[t, :]
            xtilde_t = self.decoder.get_current_rnn_input(
                E_tm1, htilde_tm1, h, F_lens)
            htilde_tm1 = self.decoder.get_current_hidden_state(
                xtilde_t, htilde_tm1)
            if self.cell_type == 'lstm':
                logits_t = self.get_current_logits(htilde_tm1[0])
            else:
                logits_t = self.get_current_logits(htilde_tm1)

            # TODO: Think about if attention is used here
            # -- it shouldn't (should be included in get_current_hidden_state already)

        logits = torch.cat(logits_list, dim=1)  # Dimension of concatenation is T
        # ? https://github.com/JoshFeldman95/translation/blob/7fba309e6914ee4235e95fa2ffd3294f7db5d6a1/models.py
        # ? https://github.com/SunSiShining/CopyRNN-Pytorch/blob/2e37cce64d44e7c99a624e7f47c1b9c57094dadf/pykp/model.py
        # ? https://github.com/sh951011/PyTorch-Seq2seq/blob/66c4c2bae6b8f432f87bf532c3004de2cee56f99/models/decoder.py
        return logits


    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        """
        For n, k, v (tm1 -> cand):
            b_cand_0 of kth path appending vocab v = htilde_t of kth path at t+1
            b_cand_1 of kth path appending vocab v = concatenate [b_tm1_1, v]
            logpb_cand = logpb_tm1_1 + logpy_t with vocab v
        For n, k (cand -> t):
            b_t = max of logpb_cand, for both b_t_1 and b_t_0
        """

        # Define some dimensions
        N, K, t = logpb_tm1.shape[0], logpb_tm1.shape[1], b_tm1_1.shape[0]
        KK = K * K  # K beams, each extended by K words -> K^2 (beam+word) pairs
        NK, NKK = N * K, N * KK  # Each batch has K^2 extended (beam+word) pairs

        # Select K words for each beam as candidates
        v_prob, v_id = logpy_t.topk(K, dim=-1) # (N, K, K)

        # Expand the candidates, find logpy_t of each
        logpb_cand = logpb_tm1.unsqueeze(2).expand(N, K, K) + v_prob  # (N, K, K)
        logpb_cand = logpb_cand.view(N, KK)  # (N, K*K)

        # Select K from the K^2 candidates
        logpb_t, cand_id=logpb_cand.topk(K, dim=-1)  # (N, K)

        # Select corresponding K words used to update the beams
        NK_steper = torch.arange(0, NKK, KK, dtype=cand_id.dtype,
                            device=cand_id.device).unsqueeze(1).expand_as(cand_id)
        cand_v_id = (cand_id + NK_steper).view(NK) # (NK, )
        v_id = v_id.view(NKK).index_select(0, cand_v_id).view(NK, 1)  # (NKK, ) -> (NK, )

        # Select corresponding K beams and update b_t
        b_t_id = (cand_id / K + N_steper).view(NK) # (NK, )
        b_t_0 = htilde_t.view(NK, -1).index_select(0, b_t_id).view(N, K, -1)  # (N, K, 2H) -> (NK, 2H) -> (N, K, 2H)
        b_tm1_selected = b_tm1_1.view(t, NK).index_select(1, b_t_id) # (t, N, K) -> (t, NK)
        b_t_1 = torch.cat((b_tm1_selected, v_id), -1).view(t+1, N, K) # (t+1, NK) -> (t+1, N, K)

        return b_t_0, b_t_1, logpb_t
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # logpy_t is of shape (N, K, V)

        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)

        # relevant pytorch modules:
        # torch.{flatten,unsqueeze,expand_as,gather,cat}
