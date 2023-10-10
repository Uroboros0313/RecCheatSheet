import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import DNN


class MMOEBlock(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

    def multi_gate_scores(self):
        pass

    def multi_expert_embedding(self):
        pass

    def task_specified_tower_forward(self):
        pass

    def forward(self):
        pass


class PLEBlock(nn.Module):
    def __init__(self):
        super().__init__()


class MVKEBlock(nn.Module):
    def __init__(self,
                 num_feats,
                 num_kernel=4,
                 hidden_dims=[32, 32, 32],
                 emb_dim=32,
                 combiner='sum',
                 dropout_rate=0.3):
        super().__init__()

        self.num_feats = num_feats
        self.num_kernel = num_kernel
        self.emb_dim = emb_dim
        self.combiner = combiner
        self.dropout_rate = dropout_rate

        self.vk_embs = torch.nn.parameter.Parameter(torch.zeros((emb_dim, num_kernel)))
        self.__reset_parameters()

        if combiner == 'sum':
            input_dim = emb_dim
        elif combiner == 'concat':
            input_dim = emb_dim * num_feats

        self.vk_towers = nn.ModuleList(
            [DNN(input_dim, hidden_dims) for _ in range(self.num_kernel)])

    def __reset_parameters(self):
        nn.init.xavier_uniform_(self.vk_embs.data, gain=nn.init.calculate_gain('relu'))

    def virtual_kernel_gate_probs(self, tag_embedding):
        vk_gate_scores = torch.mm(tag_embedding, self.vk_embs) \
                         / torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float32))  # [BATCH_SIZE, NUM_KERNELS]

        vk_gate_probs = torch.softmax(vk_gate_scores, dim=-1).unsqueeze(1)  # [BATCH_SIZE, 1, NUM_KERNELS]
        vk_gate_probs = F.dropout(vk_gate_probs, self.dropout_rate, training=self.training)

        return vk_gate_probs

    def virtual_kernel_expert_embeddings(self, x):
        vk_scores = torch.matmul(x, self.vk_embs)
        vk_probs = torch.softmax(vk_scores.transpose(-1, -2), dim=-1)  # [BATCH_SIZE, NUM_KERNELS, NUM_FEATS]
        vk_probs = F.dropout(vk_probs, self.dropout_rate, training=self.training)

        if self.combiner == 'sum':
            context = torch.matmul(vk_probs, x) \
                      / torch.sqrt(
                torch.tensor(self.emb_dim, dtype=torch.float32))  # [BATCH_SIZE, NUM_KERNELS, EMB_DIM]

        elif self.combiner == 'concat':
            vk_probs = vk_probs.unsqueeze(3)  # [BATCH_SIZE, NUM_KERNELS, NUM_FEATS, 1]
            x = x.unsqueeze(1)  # [BATCH_SIZE, 1, NUM_FEATS, EMB_DIM]
            context = (vk_probs * x) \
                .reshape(-1, self.num_kernel,
                         self.num_feats * self.emb_dim)  # [BATCH_SIZE, NUM_KERNELS, NUM_FEATS * EMB_DIM]

        return context

    def virtual_kernel_expert_outputs(self, context):
        vk_outputs = []
        for i in range(self.num_kernel):
            vk_outputs.append(self.vk_towers[i](context[:, i, :]).unsqueeze(1))

        vk_outputs = torch.cat(vk_outputs, dim=1)

        return vk_outputs

    def forward(self, x, tag_embedding):
        '''
        param x: input data x, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        param tag_embedding: input data tag_embedding, shape (BATCH_SIZE, EMB_DIM), dtype torch.float32
        
        return: output data, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        '''
        context = self.virtual_kernel_expert_embeddings(x)
        vk_outputs = self.virtual_kernel_expert_outputs(context)
        vk_gate_probs = self.virtual_kernel_gate_probs(tag_embedding)

        user_embedding = torch.matmul(vk_gate_probs, vk_outputs).squeeze(1)

        return user_embedding


if __name__ == '__main__':
    def test_mvke():
        mock_ninput = torch.ones((64, 10, 32)).float()
        mock_tag = torch.randn((64, 32))
        model = MVKEBlock(10, combiner='concat')
        print(model(mock_ninput, mock_tag).shape)


    test_mvke()
