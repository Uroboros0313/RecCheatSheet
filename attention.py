import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 attn_head_dim,
                 num_heads=4,
                 dropout_rate=0.2,
                 pooling_type=None
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type

        self.Q = nn.Linear(input_dim, attn_head_dim * num_heads)
        self.K = nn.Linear(input_dim, attn_head_dim * num_heads)
        self.V = nn.Linear(input_dim, attn_head_dim * num_heads)

    def attention_head_transform(self, mat):
        batch_size, seq_len, _ = mat.shape

        mat = mat.view(batch_size,
                       seq_len,
                       self.num_heads,
                       self.attn_head_dim)

        return mat.permute(0, 2, 1, 3)

    def scaled_dot_product_attention_probs(self, queries, keys, mask):
        # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN]
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) \
                      / torch.sqrt(torch.tensor(self.attn_head_dim, dtype=torch.float32))
        attn_scores = attn_scores + mask
        attn_probs = torch.softmax(attn_scores, dim=-1)

        return attn_probs

    def forward(self, x, mask=None):
        '''
        param x: input data x, shape (BATCH_SIZE, SEQ_LEN, EMB_DIM), dtype torch.float32
        param mask: attention mask, shape (BATCH_SIZE, SEQ_LEN), dtype torch.float32
        
        return: output data, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        '''
        batch_size, seq_len, _ = x.shape

        if mask is None:
            mask = torch.ones((batch_size, 1, 1, seq_len))
        else:
            mask = (1.0 - mask) * -1e4

        # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, ATTN_HEAD_DIM]
        queries = self.attention_head_transform(self.Q(x))
        keys = self.attention_head_transform(self.K(x))
        values = self.attention_head_transform(self.V(x))

        attn_probs = self.scaled_dot_product_attention_probs(queries, keys, mask)
        attn_probs = F.dropout(attn_probs, p=self.dropout_rate, training=self.training)
        context = torch.matmul(attn_probs, values) \
            .permute(0, 2, 3, 1) \
            .contiguous()

        if self.pooling_type == 'mean' or self.pooling_type is None:
            attn_out = torch.mean(context, dim=-1)
        elif self.pooling_type == 'concat':
            attn_out = context.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Pooling Method `{self.pooling_type}` not Exists")

        return attn_out


if __name__ == '__main__':
    def test_mhsa():
        num_heads = 8
        emb_dim = 32
        batch_size = 64
        num_feats = 20

        mock_input = torch.randn((batch_size, num_feats, emb_dim))
        mhsa = MultiHeadSelfAttentionBlock(emb_dim, emb_dim, num_heads, pooling_type='concat')

        print(mhsa(mock_input).shape)


    test_mhsa()
