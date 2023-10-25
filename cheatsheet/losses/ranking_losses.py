import tensorflow as tf

from math_utils import tf_log2


class BprLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, logits, uid_tensor, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, eps)

    def __loss_fn__(self, y_true, logits, uid_tensor, eps):
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)
        label_mask = tf.cast(tf.not_equal(y_true, tf.transpose(y_true, perm=[1, 0])), tf.float32)

        bpr_label_mat = (tf.subtract(y_true, tf.transpose(y_true, perm=[1, 0])) + 1.0) / 2.0
        bpr_logits_mat = tf.sigmoid(tf.subtract(logits, tf.transpose(logits, perm=[1, 0])))

        loss_mat = bpr_label_mat * tf.math.log(bpr_logits_mat + eps) + \
                   (1.0 - bpr_label_mat) * tf.math.log(1.0 - bpr_logits_mat + eps)
        loss_ = - tf.reduce_sum(label_mask * uid_mask * loss_mat)

        return loss_


class InBatchSoftMaxLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return self.__loss_fn__(x)

    def __loss_fn__(self, x):
        return super().__loss_fn__(x)


class DistillRankingLoss():
    def __init__(self, use_dvalue=False) -> None:
        self.use_dvalue = use_dvalue

    def __call__(self, y_true, logits, uid_tensor, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, eps)

    def __loss_fn__(self, y_true, logits, uid_tensor, eps):
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)
        dr_label_mat = tf.cast(tf.greater(y_true, tf.transpose(y_true, perm=[1, 0])), tf.float32)
        dr_logits_mat = tf.sigmoid(tf.subtract(logits, tf.transpose(logits, perm=[1, 0])))

        loss_mat = dr_label_mat * tf.math.log(dr_logits_mat + eps) + \
                   (1.0 - dr_label_mat) * tf.math.log(1.0 - dr_logits_mat + eps)

        if self.use_dvalue:
            dvalue_mat = tf.abs(tf.subtract(y_true, tf.transpose(y_true, perm=[1, 0])))
            loss_mat = loss_mat * dvalue_mat

        loss_ = - tf.reduce_sum(uid_mask * loss_mat)

        return loss_


class BucketDistillRankingLoss(DistillRankingLoss):
    def __init__(self, boundaries=[0.2, 0.4, 0.6, 0.8], use_dvalue=False):
        self.boundaries = sorted(boundaries)
        self.num_bucket = len(boundaries) + 1
        self.bucketizer = tf.raw_ops.Bucketize

        super().__init__(use_dvalue=use_dvalue)

    def __bucketize__(self, y_true):
        bucket_idxs = self.bucketizer(input=y_true, boundaries=self.boundaries)
        norm_bucketized_values = \
            tf.divide(tf.cast(bucket_idxs, tf.float32), self.num_bucket)

        return norm_bucketized_values

    def __call__(self, y_true, logits, uid_tensor, eps=1e-4):
        bucketized_y_true = self.__bucketize__(y_true)
        return self.__loss_fn__(bucketized_y_true, logits, uid_tensor, eps)


class ListMLELoss():
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, logits, uid_tensor=None, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, eps)

    def __loss_fn__(self, y_true, logits, uid_tensor, eps):
        indices = tf.argsort(y_true, axis=0, direction='DESCENDING')
        indices = tf.squeeze(indices, axis=-1)

        sorted_logits = tf.math.exp(tf.gather(logits, indices))
        sorted_uid = tf.gather(uid_tensor, indices)

        uid_mask = tf.cast(tf.equal(sorted_uid, tf.transpose(sorted_uid, perm=[1, 0])), tf.float32)
        logits_mat = tf.transpose(sorted_logits, perm=[1, 0]) + tf.zeros_like(sorted_logits, dtype=tf.float32)

        masked_logits_mat = tf.linalg.band_part(logits_mat, 0, -1)
        masked_logits_sum = tf.reduce_sum(tf.multiply(masked_logits_mat, uid_mask), axis=-1, keepdims=True)
        masked_logits_norm = tf.divide(sorted_logits, masked_logits_sum)
        loss_ = - tf.reduce_sum(tf.math.log(masked_logits_norm + eps))

        return loss_


class ListNetLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, logits, uid_tensor, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, eps)

    def __loss_fn__(self, y_true, logits, uid_tensor, eps):
        logits = tf.math.exp(logits)
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)
        logits_mat = tf.transpose(logits, perm=[1, 0]) + tf.zeros_like(logits, dtype=tf.float32)

        masked_logits_sum = tf.reduce_sum(tf.multiply(uid_mask, logits_mat), axis=-1, keepdims=True)
        masked_logits_norm = tf.divide(logits, masked_logits_sum)

        indices = tf.squeeze(tf.greater(y_true, 0.5))
        loss_arr = - tf.math.log(masked_logits_norm + eps)
        masked_loss_arr = tf.boolean_mask(loss_arr, tf.squeeze(indices))
        loss_ = tf.reduce_sum(masked_loss_arr)

        return loss_


class MetricWeight():
    def __init__(self, metric='LambdaRank'):
        self.__metric_fn__ = (lambda **kwargs: 1.0)
        self.metric = metric

        if metric == "LambdaRank":
            self.__metric_fn__ = self.lambda_rank_weight
        elif metric == "ndcgCost1":
            self.__metric_fn__ = self.ndcg_cost1_weight
        elif metric == "ndcgCost2":
            self.__metric_fn__ = self.ndcg_cost2_weight
        elif metric == "ndcgCostMix":
            self.__metric_fn__ = self.ndcg_cost_mix_weight
        elif metric == "arpCost1":
            self.__metric_fn__ = self.arp_cost1
        elif metric == "arpCost2":
            self.__metric_fn__ = self.arp_cost2
        else:
            pass

    def __call__(self, **kwargs):
        return self.__metric_fn__(**kwargs)

    @staticmethod
    def lambda_rank_weight(**kwargs):
        G, D = kwargs.get("G", None), kwargs.get("D", None)

        rawinv_d = tf.pow(D, -1.0)
        inf_mask = tf.math.is_finite(rawinv_d)
        inv_d = tf.where(inf_mask, rawinv_d, 0.0)

        diff_g_mat = tf.abs(G - tf.transpose(G, perm=[1, 0]))
        diff_inv_d_mat = tf.abs(inv_d - tf.transpose(inv_d, perm=[1, 0]))

        weights = diff_g_mat * diff_inv_d_mat
        return weights

    @staticmethod
    def ndcg_cost1_weight(**kwargs):
        G, D = kwargs.get("G", None), kwargs.get("D", None)

        raw_weights = tf.transpose(G / D, perm=[1, 0])
        nan_mask = tf.math.is_nan(raw_weights)

        weights = tf.where(condition=~nan_mask, x=raw_weights, y=0.0)
        return weights

    @staticmethod
    def ndcg_cost2_weight(**kwargs):
        G, D = kwargs.get("G", None), kwargs.get("D", None)

        abs_diff_g = tf.abs(G - tf.transpose(G, perm=[1, 0]))
        pos_mat = tf.cumsum(tf.ones_like(D, dtype=tf.float32), axis=-1)
        rel_pos_mat = tf.abs(pos_mat - tf.transpose(pos_mat, perm=[1, 0]))
        inv_delta_pos_g = tf.abs(tf.pow(tf_log2(rel_pos_mat + 1), -1) - tf.pow(tf_log2(rel_pos_mat + 2), -1))

        null_mask = tf.math.is_nan(inv_delta_pos_g) | tf.math.is_inf(inv_delta_pos_g)

        weights = tf.where(condition=(~null_mask) & (G > 0.0),
                           x=inv_delta_pos_g * abs_diff_g,
                           y=0.0)

        return weights

    @staticmethod
    def ndcg_cost_mix_weight(**kwargs):
        pass

    @staticmethod
    def arp_cost1(**kwargs):
        y_true = kwargs.get("y_true", None)
        abs_weight = tf.math.abs(y_true)
        weights = tf.transpose(abs_weight, perm=[1, 0]) + tf.zeros_like(abs_weight)

        return weights

    @staticmethod
    def arp_cost2(**kwargs):
        y_true = kwargs.get("y_true", None)

        score_diff = y_true - tf.transpose(y_true, perm=[1, 0])
        weights = tf.abs(score_diff)

        return weights


class LambdaLoss():
    def __init__(self, k=None, metric='LambdaRank') -> None:
        '''
        param k: metric@K
        param metric: {'ndcgCost1', 'LambdaRank'}
        '''
        self.k = k  # NDCG@K
        self.metric = metric
        self.metric_weight_fn = MetricWeight(metric=metric)

    def __call__(self, y_true, logits, uid_tensor, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, eps)

    def __loss_fn__(self, y_true, logits, uid_tensor, eps):
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)

        # uid内部排序
        gap_values = tf.reduce_sum(uid_mask, axis=-1, keepdims=True)
        gap_y_true = y_true - gap_values
        gap_y_pred = tf.sigmoid(logits) - gap_values

        true_rank = tf.squeeze(tf.argsort(gap_y_true, axis=0, direction="DESCENDING"), axis=-1)
        pred_rank = tf.squeeze(tf.argsort(gap_y_pred, axis=0, direction="DESCENDING"), axis=-1)

        y_true_sorted = tf.gather(y_true, indices=true_rank)
        true_sorted_by_pred = tf.gather(y_true, indices=pred_rank)
        y_pred_sorted = tf.gather(logits, indices=pred_rank)

        D = tf.where(condition=tf.cast(uid_mask, dtype=tf.bool),
                     x=tf_log2(1. + tf.cumsum(uid_mask, axis=1)),
                     y=uid_mask)
        best_dcg = tf.where(condition=tf.cast(uid_mask, dtype=tf.bool),
                           x=(tf.math.pow(2.0, tf.transpose(y_true_sorted, perm=[1, 0]) - 1.0)) / D,
                           y=uid_mask)
        inv_max_dcg = 1.0 / tf.reduce_sum(best_dcg, axis=-1, keepdims=True)
        G = tf.where(condition=tf.cast(uid_mask, dtype=tf.bool),
                     x=(tf.math.pow(2.0, tf.transpose(true_sorted_by_pred, perm=[1, 0]) - 1.0)) * inv_max_dcg,
                     y=uid_mask)

        # BPR Loss加权
        label_mat = tf.cast(tf.greater(true_sorted_by_pred,
                                       tf.transpose(true_sorted_by_pred, perm=[1, 0])), tf.float32)
        logits_mat = tf.sigmoid(tf.subtract(y_pred_sorted, tf.transpose(logits, perm=[1, 0])))
        loss_mat = label_mat * tf.math.log(logits_mat + eps) + \
                   (1.0 - label_mat) * tf.math.log(1.0 - logits_mat + eps)

        weights = self.metric_weight_fn(G=G, D=D, y_true=true_sorted_by_pred)

        # metric@K的掩码
        if self.k != None:
            topK_row_mask = tf.where(tf.cumsum(uid_mask, axis=0) <= self.k, 1.0, 0.0)
            topK_mask = topK_row_mask * tf.transpose(topK_row_mask, perm=[1, 0])
        else:
            topK_mask = tf.ones_like(loss_mat, tf.float32)

        loss_ = -tf.reduce_sum(weights * loss_mat * uid_mask * topK_mask)
        return loss_


class ApproxMetricLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, logits, uid_tensor, temperture=0.1, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, temperture, eps)

    def approx_rank(self, uid_mask, logits, temperture):
        uid_mask = uid_mask - tf.compat.v1.matrix_diag(tf.linalg.diag_part(uid_mask))
        logits_mat = tf.sigmoid((logits - tf.transpose(logits, perm=[1, 0])) / temperture)
        appr_rank = tf.reduce_sum(input_tensor=tf.transpose(uid_mask * logits_mat, perm=[1, 0]),
                                  axis=-1,
                                  keepdims=True) + 1.0

        return appr_rank

    def __loss_fn__(self, **kwargs):
        raise NotImplementedError("Lossfn not implemented")


class ApproxNDCGLoss(ApproxMetricLoss):
    def __init__(self) -> None:
        pass

    def __loss_fn__(self, y_true, logits, uid_tensor, temperture, eps):
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)

        gap_values = tf.reduce_sum(uid_mask, axis=-1, keepdims=True)
        gap_y_true = y_true - gap_values

        true_rank = tf.squeeze(tf.argsort(gap_y_true, axis=0, direction="DESCENDING"), axis=-1)
        y_true_sorted = tf.gather(y_true, indices=true_rank)

        D = tf.where(condition=tf.cast(uid_mask, dtype=tf.bool),
                     x=tf_log2(1.0 + tf.cumsum(uid_mask, axis=1)),
                     y=uid_mask)
        best_dcg = tf.where(condition=tf.cast(uid_mask, dtype=tf.bool),
                           x=(tf.math.pow(2.0, tf.transpose(y_true_sorted, perm=[1, 0]) - 1.0)) / D,
                           y=uid_mask)
        inv_max_dcg = 1.0 / tf.reduce_sum(best_dcg, axis=-1, keepdims=True)

        appr_rank = self.approx_rank(uid_mask, logits, temperture)
        appr_discount = 1.0 / tf_log2(1.0 + appr_rank)
        G = tf.pow(2.0, y_true) - 1.0

        loss_ = -tf.reduce_sum(G * appr_discount * inv_max_dcg)
        return loss_


class ApproxMRRLoss(ApproxMetricLoss):
    def __init__(self) -> None:
        super().__init__()

    def __loss_fn__(self, y_true, logits, uid_tensor, temperture, eps):
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)
        appr_rank = self.approx_rank(uid_mask, logits, temperture)
        appr_discount = 1.0 / tf_log2(1.0 + appr_rank)

        loss_ = - tf.reduce_sum(appr_discount * y_true)
        return loss_
    


if __name__ == '__main__':
    uid_tensor = tf.reshape(tf.constant([1, 1, 1, 4, 4, 4, 4]), [-1, 1])
    true_ = tf.reshape(tf.cast([0, 1, 1, 0, 0, 0, 1], dtype=tf.float32), [-1, 1])
    rels = tf.reshape(tf.constant([0.01, 0.9, 0.87, 0.12, 0.10, 0.13, 0.68]), [-1, 1])
    preds = tf.reshape(tf.constant([0.5, 0.8, 0.2, 0.01, 0.23, 0.4, 0.1]), [-1, 1]) * 2 - 1

    print(f"ListMLELoss:{ListMLELoss()(rels, preds, uid_tensor)}")
    print(f"ListNetLoss:{ListNetLoss()(true_, preds, uid_tensor)}")
    print(f"BprLoss:{BprLoss()(true_, preds, uid_tensor)}")
    print(f"DistillRankingLoss:{DistillRankingLoss()(rels, preds, uid_tensor)}")
    print(f"WeightedDistillRankingLoss:{DistillRankingLoss(use_dvalue=True)(rels, preds, uid_tensor)}")
    print(f"BucketDistillRankingLoss:{BucketDistillRankingLoss()(rels, preds, uid_tensor)}")
    print(f"LambdaRankLoss:{LambdaLoss()(rels, preds, uid_tensor)}")
    print(f"ndcgCost1Loss:{LambdaLoss(metric='ndcgCost1')(rels, preds, uid_tensor)}")
    print(f"ndcgCost2Loss:{LambdaLoss(metric='ndcgCost2')(rels, preds, uid_tensor)}")
    print(f"arpCost1Loss:{LambdaLoss(metric='arpCost1')(rels, preds, uid_tensor)}")
    print(f"arpCost2Loss:{LambdaLoss(metric='arpCost2')(rels, preds, uid_tensor)}")
    print(f"apporxNDCG:{ApproxNDCGLoss()(rels, preds, uid_tensor)}")
    print(f"approxMRR:{ApproxMRRLoss()(rels, preds, uid_tensor)}")
