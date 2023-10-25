from typing import List, Dict, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..layers.core import MLP, Linear, BatchNormalization
from ..layers.mixture import MMOEBlock


def MMOE(num_expert:int, expert_emb_dim:int, expert_hidden_units:List[int]=[128, 64], 
         gate_hidden_units:List[int]=[32, 16], casade_routing:List[Tuple[int, int]]=[], casade_gradient:bool=False,
         use_bn:bool=False, dropout_rate:float=0.3):
     '''
     param num_expert: 专家网络数量
     param exper_emb_dim: 专家网络输出embedding大小
     param expert_hidden_units: 专家网络隐藏层尺寸
     param gate_hidden_units: 门控网络隐藏层尺寸
     param casade_routing: 相关目标embedding拼接的路径, [(1, 2)]代表专家1的输出拼接到专家2的输出
     param casade_gradient: 继承embedding是否需要截断梯度反传
     param use_bn: 是否使用BatchNormalization
     param dropout_rate: 门控网络、专家网络、防止gate极化的dropout概率
     '''
     
     '''
     # MOCK
     task_embs = MMOE()(input)
     task_casade_list = [[task_emb] for task_emb in task_embs]
     for src, tgt in casade_routing:
          src_embs = task_casade_list[src][0]
          if not casade_gradient:
               src_embs = tf.stop_gradient(task_casade_list[src][0]) 
               
          task_casade_list[tgt].append(src_embs)
          
     task_dnn_inputs = [tf.squeeze(tf.concat(task_casade_embs, axis=2), axis=-1) 
          for task_casade_embs in task_casade_list]
     '''
     pass