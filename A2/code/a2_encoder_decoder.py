# # Copyright 2020 University of Toronto, all rights reserved

# '''Concrete implementations of abstract base classes.

# You don't need anything more than what's been imported here
# '''

# import torch

# from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


# class Encoder(EncoderBase):

#     def init_submodules(self):
#         '''Initialize the parameterized submodules of this network'''

#         '''embedding : extracts learned token embeddings for each index in a token sequence, watch out for padding'''
#         self.embedding = torch.nn.Embedding(self.source_vocab_size,
#                                             self.word_embedding_size,
#                                             padding_idx=self.pad_id)

#         '''rnn : rnn layer that processes source word embeddings, bidirectional'''
#         if self.cell_type == 'lstm':
#             self.rnn = torch.nn.LSTM(self.word_embedding_size,
#                                      self.hidden_state_size,
#                                      self.num_hidden_layers,
#                                      dropout=self.dropout,
#                                      bidirectional=True)
#         elif self.cell_type == 'gru':
#             self.rnn = torch.nn.GRU(self.word_embedding_size,
#                                     self.hidden_state_size,
#                                     self.num_hidden_layers,
#                                     dropout=self.dropout,
#                                     bidirectional=True)
#         elif self.cell_type == 'rnn':
#             self.rnn = torch.nn.RNN(self.word_embedding_size,
#                                     self.hidden_state_size,
#                                     self.num_hidden_layers,
#                                     dropout=self.dropout,
#                                     bidirectional=True)

#     def get_all_rnn_inputs(self, F):
#         '''compute input vectors for each source transcription'''
#         x = self.embedding(F).to(F.device)

#         # Mask x by seting to zeros if "padded"
#         x[(F == self.pad_id).nonzero(as_tuple=True)] = 0

#         return x  # (S, N) -> (S, N, I)

#     def get_all_hidden_states(self, x, F_lens, h_pad):
#         # Pack hidden states
#         x_packed = torch.nn.utils.rnn.pack_padded_sequence(
#             x, F_lens, enforce_sorted=False)
#         # compute all final hidden states for provided input sequence.
#         outputs, hidden = self.rnn(x_packed)
#         # Unpack hidden states
#         outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
#             outputs, padding_value=h_pad, total_length=x.shape[0])

#         # print("-----\nIn get_all_hidden_states, x: {} -> h: {}\n-----".format(x.shape, outputs.shape))
#         return outputs.to(x.device)  # x: (S, N, I) -> outputs: (S, N, 2 * H)


# class DecoderWithoutAttention(DecoderBase):
#     '''A recurrent decoder without attention'''

#     def init_submodules(self):
#         ''' Embedding '''
#         self.embedding = torch.nn.Embedding(self.target_vocab_size,
#                                             self.word_embedding_size,
#                                             padding_idx=self.pad_id)

#         ''' RNN '''
#         input_size = self.word_embedding_size # + self.hidden_state_size
#         hidden_size = self.hidden_state_size

#         if self.cell_type == 'lstm':
#             self.cell = torch.nn.LSTMCell(input_size, hidden_size)
#         elif self.cell_type == 'gru':
#             self.cell = torch.nn.GRUCell(input_size, hidden_size)
#         elif self.cell_type == 'rnn':
#             self.cell = torch.nn.RNNCell(input_size, hidden_size)

#         ''' Fully-connected '''
#         self.ff = torch.nn.Linear(in_features=hidden_size,
#                                   out_features=self.target_vocab_size)

#     def get_first_hidden_state(self, h, F_lens):
#         '''Get the initial decoder hidden state, prior to the first input'''
#         hidden_state_mid = self.hidden_state_size // 2

#         # Take each hidden state at F_len[n-1]
#         forward = h[-1, F_lens, :hidden_state_mid]
#         backward = h[0, F_lens, hidden_state_mid:]

#         # Concatenate them together
#         htilde_tm1 = torch.cat((forward, backward), dim=1).squeeze(1)

#         # print("-----\nIn get_first_hidden_state, h: {} -> htilde_tm1: {}".format(h.shape, htilde_tm1.shape))
#         # print("-- vs. expected dimension: h: (S, N, {}) -> htilde_tm1: (N, {})\n-----".format(
#         #     self.hidden_state_size * 2, self.hidden_state_size * 2))
#         return htilde_tm1.to(h.device)  # h: (S, N, 2 * H) -> htilde_tm1: (N, 2 * H)

#     def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
#         '''Get masked embedding input and concatenate with the previous hidden state to be the new input'''
#         xtilde_t = self.embedding(E_tm1).to(h.device)
#         xtilde_t[(E_tm1 == self.pad_id).nonzero().flatten()] = 0

#         # print("-----In get_current_rnn_input, htilde_tm1: {} -> xtilde_t: {}".format(
#         #     htilde_tm1.shape, xtilde_t.shape))
#         return xtilde_t

#     def get_current_hidden_state(self, xtilde_t, htilde_tm1):
#         '''Calculate the decoder's current hidden state
#         Converts `E_tm1` to embeddings, and feeds those embeddings into
#         the recurrent cell alongside `htilde_tm1`.'''
#         if self.cell_type == 'lstm':
#             htilde_tm1 = (htilde_tm1[0][:, :self.hidden_state_size],
#                           htilde_tm1[1][:, :self.hidden_state_size])
#         else:
#             htilde_tm1 = htilde_tm1[:, :self.hidden_state_size]

#         htilde_t = self.cell(xtilde_t, htilde_tm1)
#         return htilde_t

#     def get_current_logits(self, htilde_t):
#         '''Calculate un-normalized log-probability distribution over output tokens for current time step'''
#         logits = self.ff(htilde_t)

#         # print("-----\nIn get_current_logits, htilde_tm1: {} -> logits: {}\n-----".format(
#         #     htilde_t.shape, logits.shape))
#         return logits.to(htilde_t.device) # htilde_t: (N, 2 * H) -> logits_t: (N, V)


# class DecoderWithAttention(DecoderWithoutAttention):
#     '''A decoder, this time with attention

#     Inherits from DecoderWithoutAttention to avoid repeated code.
#     '''

#     def init_submodules(self):
#         ''' Embedding '''
#         self.embedding = torch.nn.Embedding(self.target_vocab_size,
#                                             self.word_embedding_size,
#                                             padding_idx=self.pad_id)

#         ''' RNN '''
#         input_size = self.word_embedding_size  + self.hidden_state_size
#         hidden_size = self.hidden_state_size

#         if self.cell_type == 'lstm':
#             self.cell = torch.nn.LSTMCell(input_size, hidden_size)
#         elif self.cell_type == 'gru':
#             self.cell = torch.nn.GRUCell(input_size, hidden_size)
#         elif self.cell_type == 'rnn':
#             self.cell = torch.nn.RNNCell(input_size, hidden_size)

#         ''' Fully-connected '''
#         self.ff = torch.nn.Linear(in_features=hidden_size,
#                                   out_features=self.target_vocab_size)

#     def get_first_hidden_state(self, h, F_lens):
#         '''same as before, but initialize to zeros'''
#         # h : (S, N, self.hidden_state_size) -> htilde_0 : (N, self.hidden_state_size)
#         return torch.zeros_like(h[-1], dtype=h.dtype, device=h.device)

#     def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
#         '''Update to account for attention. Use attend() for c_t'''
#         embedded = self.embedding(E_tm1)
#         embedded[(E_tm1 == self.pad_id).nonzero().flatten()] = 0

#         # If lstm, take only the hidden state (instead of the cell state)
#         if self.cell_type == 'lstm':
#             attention = self.attend(htilde_tm1[0], h, F_lens)
#         else:
#             attention = self.attend(htilde_tm1, h, F_lens)

#         # Concatenate the word input and attention input to be the new input
#         xtilde_t = torch.cat((embedded, attention), dim=1)
#         return xtilde_t

#     def attend(self, htilde_t, h, F_lens):
#         '''Compute context vector c_t'''
#         alphas = self.get_attention_weights(
#             htilde_t, h, F_lens).transpose(0, 1).unsqueeze(1) # (N, 1, S)
#         h = h.permute(1, 0, 2)  # (N, S, 2*H)
#         c_t = torch.bmm(alphas, h).squeeze(1)  # (N, 2 * H)
#         return c_t

#     def get_attention_weights(self, htilde_t, h, F_lens):
#         # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
#         # in h have weight 0 and no gradient. You have to implement
#         # get_energy_scores()
#         # alpha_t (output) is of shape (S, N)
#         e_t = self.get_energy_scores(htilde_t, h)
#         pad_mask = torch.arange(h.shape[0], device=h.device)
#         pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
#         e_t = e_t.masked_fill(pad_mask, -float('inf'))
#         return torch.nn.functional.softmax(e_t, 0)

#     def get_energy_scores(self, htilde_t, h):
#         '''Determine energy scores via cosine similarity'''
#         S = h.shape[0]
#         # e_t = torch.zeros(S, N, device=h.device)
#         # for s in range(S):
#         #     e_t[s] = torch.nn.functional.cosine_similarity(
#         #         htilde_t, h[s], dim=1)

#         energy_func = torch.nn.CosineSimilarity(dim=2)
#         # htilde_t: (N, 2 * H) -> h: (S, N, 2 * H)
#         e_t = energy_func(htilde_t.unsqueeze(0).repeat(S, 1, 1), h)
#         return e_t  # (S, N)


# class EncoderDecoder(EncoderDecoderBase):

#     def init_submodules(self, encoder_class, decoder_class):
#         self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
#                                      pad_id=self.source_pad_id,
#                                      word_embedding_size=self.word_embedding_size,
#                                      num_hidden_layers=self.encoder_num_hidden_layers,
#                                      hidden_state_size=self.encoder_hidden_size,
#                                      dropout=self.encoder_dropout,
#                                      cell_type=self.cell_type)
#         decoder_hidden_size = self.encoder_hidden_size*2
#         self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
#                                      pad_id=self.source_pad_id,
#                                      word_embedding_size=self.word_embedding_size,
#                                      hidden_state_size=decoder_hidden_size,
#                                      cell_type=self.cell_type)

#     def get_logits_for_teacher_forcing(self, h, F_lens, E):
#         '''get logits over entire E. logits predict the *next* word in the sequence.'''

#         T = E.shape[0]

#         # Initialize output results (i.e., just logits here, no token needed)
#         logits_list = []

#         # Initialize logit and hidden state corresponding to <SOS>
#         logit, h_tilde = self.decoder.forward(E[0], None, h, F_lens)

#         # Loop through the sequence(s) with E[t] as the input word (as opposed to output at t)
#         for t in range(T-1):
#             logit, h_tilde = self.decoder(E[t], h_tilde, h, F_lens) # Each logits: (N, V)
#             logits_list.append(logit)

#         # Dimension of concatenation is T
#         logits = torch.stack(logits_list, dim=0)

#         # print("-----\nIn get_logits_for_teacher_forcing, h: {} -> logits: {}\n-----".format(
#         #     h.shape, logits.shape)) # current: 100, 320000 = N, T-1*Vo
#         return logits  # E: (T, N) + h: (S, N, 2 * H) -> logits: (T - 1, N, V)

#     def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
#         """
#         For n, k, v (tm1 -> cand):
#             b_cand_0 of kth path appending vocab v = htilde_t of kth path at t+1
#             b_cand_1 of kth path appending vocab v = concatenate [b_tm1_1, v]
#             logpb_cand = logpb_tm1_1 + logpy_t with vocab v
#         For n, k (cand -> t):
#             b_t = max of logpb_cand, for both b_t_1 and b_t_0
#         """

#         # # Define some dimensions
#         # N, K, t = logpb_tm1.shape[0], logpb_tm1.shape[1], b_tm1_1.shape[0]
#         # # K beams, each extended by K words -> K^2 (beam+word) pairs
#         # KK = K * K
#         # # Each batch has K^2 extended (beam+word) pairs
#         # NK, NKK = N * K, N * KK

#         # # Select K words for each beam as candidates
#         # v_prob, v_id = logpy_t.topk(K, dim=-1)  # (N, K, K)

#         # # Expand the candidates, find logpy_t of each
#         # logpb_cand = logpb_tm1.unsqueeze(2).expand(
#         #     N, K, K) + v_prob  # (N, K, K)
#         # logpb_cand = logpb_cand.view(N, KK)  # (N, K*K)

#         # # Select K from the K^2 candidates
#         # logpb_t, cand_id = logpb_cand.topk(K, dim=-1)  # (N, K)

#         # # Select corresponding K words used to update the beams
#         # NK_steper = torch.arange(0, NKK, KK, dtype=cand_id.dtype,
#         #                          device=cand_id.device).unsqueeze(1).expand_as(cand_id)
#         # cand_v_id = (cand_id + NK_steper).view(NK)  # (NK, )
#         # v_id = v_id.view(NKK).index_select(
#         #     0, cand_v_id).view(NK, 1)  # (NKK, ) -> (NK, )

#         # # Select corresponding K beams and update b_t
#         # N_steper = torch.arange(0, NK, K, dtype=cand_id.dtype,
#         #                         device=cand_id.device).unsqueeze(1).expand_as(cand_id)
#         # b_t_id = (cand_id / K + N_steper).view(NK)  # (NK, )
#         # # (N, K, 2H) -> (NK, 2H) -> (N, K, 2H)
#         # b_t_0 = htilde_t.view(NK, -1).index_select(0, b_t_id).view(N, K, -1)
#         # b_tm1_selected = b_tm1_1.reshape(t, NK).index_select(
#         #     1, b_t_id)  # (t, N, K) -> (t, NK)
#         # # print("Dimensions -- b_tm1_selected: {}, v_id: {}, N: {}, K:{}".format(
#         # #     b_tm1_selected.shape, v_id.reshape(t, NK).shape, N, K))
#         # # (t+1, NK) -> (t+1, N, K)
#         # b_t_1 = torch.cat(
#         #     (b_tm1_selected, v_id.reshape(t, NK)), -1).view(t+1, N, K)

#         # return b_t_0, b_t_1, logpb_t

#         V = logpy_t.size()[-1]
#         all_paths = logpb_tm1.unsqueeze(-1) + logpy_t  # (N, K, V), add logprobs for new extensions
#         all_paths = all_paths.view((all_paths.shape[0], -1))  # (N, K*V)
#         logpb_t, v = all_paths.topk(self.beam_width,
#                                     -1,
#                                     largest=True,
#                                     sorted=True)  # take beam_width best possible extensions
#         logpb_t = logpb_t  # (N, K)
#         # v is (N, K)
#         # v are the indices of the maximal values.
#         paths = torch.div(v, V)  # paths chosen to be kept
#         v = torch.remainder(v, V)  # the indices of the extended words that are kept
#         # choose the paths from b_tm1_1 that were kept in our next propogation
#         b_tm1_1 = b_tm1_1.gather(2, paths.unsqueeze(0).expand_as(b_tm1_1))
#         # choose the htdile that coorespond to the taken paths
#         if self.cell_type == 'lstm':
#           b_t_0 = (htilde_t[0].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[0])),
#                    htilde_t[1].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[1])))
#         else:
#           b_t_0 = htilde_t.gather(1, paths.unsqueeze(-1).expand_as(htilde_t))
#         v = v.unsqueeze(0)  # (1, N, K)
#         b_t_1 = torch.cat([b_tm1_1, v], dim=0)
#         return b_t_0, b_t_1, logpb_t
import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}

        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        init_packet = [self.word_embedding_size, self.hidden_state_size]
        init_kwargs = {'dropout': self.dropout,
                       'num_layers': self.num_hidden_layers,
                       'bidirectional': True}

        if self.cell_type == 'gru':
            initializer = torch.nn.GRU
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNN
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTM
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.rnn = initializer(*init_packet, **init_kwargs)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        return self.embedding(F)

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        # F_lens, perm_idx = F_lens.sort(0, descending=True)
        # _, unperm_idx = perm_idx.sort(0)
        # x = x[:, perm_idx, :]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        outputs, _ = self.rnn.forward(x)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, padding_value=h_pad)
        # outputs = outputs[unperm_idx]
        return outputs


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
        init_packet = [self.word_embedding_size,
                       self.hidden_state_size]
        if self.cell_type == 'gru':
            initializer = torch.nn.GRUCell
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNNCell
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTMCell
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.cell = initializer(*init_packet)
        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

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

        mid = self.hidden_state_size // 2
        return torch.cat([h[-1, F_lens, :mid],
                          h[0, F_lens, mid:]], dim=1).squeeze(1)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        # assert False, "Fill me"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mask = torch.where(E_tm1 == torch.tensor([self.pad_id]).to(device),
                           torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).to(device)
        xtilde_t = self.embedding(E_tm1) * mask.view(-1, 1)
        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        # assert False, "Fill me"
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1[0][:, :self.hidden_state_size],
                          htilde_tm1[1][:, :self.hidden_state_size])
        else:
            htilde_tm1 = htilde_tm1[:, :self.hidden_state_size]
        return self.cell(xtilde_t, htilde_tm1)

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        # assert False, "Fill me"
        logits = self.ff.forward(htilde_t)
        return logits


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention
    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']

          init_packet = [self.word_embedding_size+self.hidden_state_size,
                         self.hidden_state_size]
          if self.cell_type == 'gru':
              initializer = torch.nn.GRUCell
          elif self.cell_type == 'rnn':
              initializer = torch.nn.RNNCell
          elif self.cell_type == 'lstm':
              initializer = torch.nn.LSTMCell
          else:
              raise ValueError(f"cell type: '{self.cell_type}' not valid.")
          self.cell = initializer(*init_packet)
          self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                              self.word_embedding_size,
                                              padding_idx=self.pad_id)
          self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)
          # self.attn = torch.nn.Linear(self.hidden_size * 2, self.word_embedding_size)

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        return torch.zeros_like(h[-1])

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mask = torch.where(E_tm1 == torch.tensor([self.pad_id]).to(device),
                           torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).to(device)
        if self.cell_type == 'lstm':
            htilde_tm1 = htilde_tm1[0]  # take the hidden states
        return torch.cat([self.embedding(E_tm1) * mask.view(-1, 1), self.attend(htilde_tm1, h, F_lens)], 1)

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        alpha = self.get_attention_weights(htilde_t, h, F_lens)  # (S, N)
        alpha = alpha.transpose(0, 1)  # (N, S)
        alpha = alpha.unsqueeze(1)  # (N, 1, S)
        h = h.permute(1, 0, 2)  # (N, S, 2*H)
        print(alpha.size(), h.size(), htilde_t.size())
        c_t = torch.bmm(alpha, h).squeeze(1)  # (N, 2*H) as desired.
        print(c_t.size())
        return c_t

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
        # h = h.permute(1, 2, 0)  # (N, S, 2*H) so batches front
        # scale = torch.inverse(torch.sqrt(self.hidden_state_size * 2))
        # htilde = htilde_t.unsqueeze(1)  # (N, 1, 2*H)
        # energy = scale * torch.bmm(h, htilde)  # (N, S, 1)
        # energy.squeeze(2).transpose(0, 1)  # (S, N) as desired
        energy = torch.zeros(h.size()[:2], device=h.device)
        for s in range(h.size()[0]):
          energy[s] = torch.nn.functional.cosine_similarity(htilde_t,
                                                            h[s], dim=1)
        return energy

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
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
        self.encoder.init_submodules()
        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type)
        self.decoder.init_submodules()

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
        # give start of string
        # initialize the first hidden state
        logits = []  # for holding logits as we do all steps in time
        l, h_tilde_tm1 = self.decoder.forward(E[0], None, h, F_lens)  # SOS tag
        for t in range(1, E.size()[0]):  # T-1
            l, h_tilde_tm1 = self.decoder.forward(E[t], h_tilde_tm1, h, F_lens)
            logits.append(l)
        logits = torch.stack(logits, 0)
        return logits

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
        V = logpy_t.size()[-1]
        all_paths = logpb_tm1.unsqueeze(-1) + logpy_t  # (N, K, V), add logprobs for new extensions
        all_paths = all_paths.view((all_paths.shape[0], -1))  # (N, K*V)
        logpb_t, v = all_paths.topk(self.beam_width,
                                    -1,
                                    largest=True,
                                    sorted=True)  # take beam_width best possible extensions
        logpb_t = logpb_t  # (N, K)
        # v is (N, K)
        # v are the indices of the maximal values.
        paths = torch.div(v, V)  # paths chosen to be kept
        v = torch.remainder(v, V)  # the indices of the extended words that are kept
        # choose the paths from b_tm1_1 that were kept in our next propogation
        b_tm1_1 = b_tm1_1.gather(2, paths.unsqueeze(0).expand_as(b_tm1_1))
        # choose the htdile that coorespond to the taken paths
        if self.cell_type == 'lstm':
          b_t_0 = (htilde_t[0].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[0])),
                   htilde_t[1].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[1])))
        else:
          b_t_0 = htilde_t.gather(1, paths.unsqueeze(-1).expand_as(htilde_t))
        v = v.unsqueeze(0)  # (1, N, K)
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        return b_t_0, b_t_1, logpb_t