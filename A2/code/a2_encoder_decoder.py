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
                               batch_first=True,
                               bidirectional=True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(self.word_embedding_size,
                              self.hidden_state_size,
                              self.num_hidden_layers,
                              dropout=self.dropout,
                              batch_first=True,
                               bidirectional=True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(self.word_embedding_size,
                              self.hidden_state_size,
                              self.num_hidden_layers,
                              dropout=self.dropout,
                              batch_first=True,
                              bidirectional=True)
        else:
            assert False, "Cell type not within provided set of types"

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        x = self.embedding(F)

        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        print("-----\nIn get_all_rnn_inputs, F: {} -> x: {}\n-----".format(F.shape, x.shape))
        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Pack hidden states
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens)
        # compute all final hidden states for provided input sequence.
        outputs, hidden = self.rnn(x_packed)
        # Unpack hidden states
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

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
            self.cell = torch.nn.LSTMCell(self.word_embedding_size + self.hidden_state_size,
                               self.hidden_state_size,
                               dropout=self.dropout,
                               batch_first=True)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size + self.hidden_state_size,
                              self.hidden_state_size,
                               dropout=self.dropout,
                               batch_first=True)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size + self.hidden_state_size,
                              self.hidden_state_size,
                              dropout=self.dropout,
                              batch_first=True)
        else:
            assert False, "Cell type not within provided set of types"

        # ff : torch.nn.Linear
        #     A fully-connected layer that converts the decoder hidden state
        #     into an un-normalized log probability distribution over target
        #     words
        self.ff = torch.nn.Linear(in_features=self.hidden_state_size,
                            out_features=self.target_vocab_size)


    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat
        assert False, "Fill me"

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        assert False, "Fill me"

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        assert False, "Fill me"

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        assert False, "Fill me"


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        assert False, "Fill me"

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        assert False, "Fill me"

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        assert False, "Fill me"

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        assert False, "Fill me"

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        assert False, "Fill me"


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
        assert False, "Fill me"

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)
        assert False, "Fill me"

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]
        assert False, "Fill me"
