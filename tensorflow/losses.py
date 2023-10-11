import tensorflow as tf


class DistillRankingLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, logits, uid_tensor, use_dvalue=False, eps=1e-4):
        return self.__loss_fn__(y_true, logits, uid_tensor, use_dvalue, eps)

    def __loss_fn__(self, y_true, logits, uid_tensor, use_dvalue, eps):
        uid_mask = tf.cast(tf.equal(uid_tensor, tf.transpose(uid_tensor, perm=[1, 0])), tf.float32)
        dr_label_mat = tf.cast(tf.greater(y_true, tf.transpose(y_true, perm=[1, 0])), tf.float32)
        dr_logits_mat = tf.sigmoid(tf.subtract(logits, tf.transpose(logits, perm=[1, 0])))

        loss_mat = dr_label_mat * tf.math.log(dr_logits_mat + eps) + \
                   (1.0 - dr_label_mat) * tf.math.log(1.0 - dr_logits_mat + eps)

        if use_dvalue:
            dvalue_mat = tf.abs(tf.subtract(y_true, tf.transpose(y_true, perm=[1, 0])))
            loss_mat = loss_mat * dvalue_mat

        loss_ = - tf.reduce_sum(uid_mask * loss_mat)

        return loss_


class BucketDistillRankingLoss(DistillRankingLoss):
    def __init__(self, boundaries=[0.2, 0.4, 0.6, 0.8]):
        self.boundaries = sorted(boundaries)
        self.num_bucket = len(boundaries) + 1
        self.bucketizer = tf.raw_ops.Bucketize

    def __bucketize__(self, y_true):
        bucket_idxs = self.bucketizer(input=y_true, boundaries=self.boundaries)
        norm_bucketized_values = tf.divide(tf.cast(bucket_idxs, tf.float32), self.num_bucket)
        return norm_bucketized_values

    def __call__(self, y_true, logits, uid_tensor, use_dvalue=False, eps=1e-4):
        bucketized_y_true = self.__bucketize__(y_true)
        return self.__loss_fn__(bucketized_y_true, logits, uid_tensor, use_dvalue, eps)


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


class LambdaLoss():
    def __init__(self, k=None) -> None:
        self.k = k  # NDCG@K

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

        # Top1的折扣系数和uid掩码
        cond_ = tf.cast(uid_mask, tf.bool) & tf.not_equal(tf.cumsum(uid_mask, axis=0), 1.0)
        discount_score = tf.where(condition=cond_,
                                  x=tf.math.log(tf.cumsum(uid_mask, axis=0) + 1.0 - tf.math.expm1(0.0)),
                                  y=uid_mask)
        discount_score = tf.reduce_sum(discount_score, axis=-1, keepdims=True) / \
                         tf.reduce_sum(uid_mask, axis=-1, keepdims=True)

        # 根据最佳排列计算IDCG
        extended_y_true_sorted = y_true_sorted + tf.zeros_like(tf.transpose(y_true_sorted, perm=[1, 0]), tf.float32)
        idcg = tf.reduce_sum(uid_mask * (extended_y_true_sorted / discount_score), axis=0)
        idcg = tf.reshape(idcg, [-1, 1])

        # Pair<i, j>交换:
        # - 损失i, j原本的 (dcg_<i, rank_i> + dcg_<j, rank_j>) / idcg
        # - 增加i, j交换的 (dcg_<i, rank_j> + dcg_<j, rank_i>) / idcg
        row_plus_dcg = tf.transpose(true_sorted_by_pred, perm=[1, 0]) / discount_score
        row_sub_dcg = true_sorted_by_pred / discount_score
        plus_dcg = row_plus_dcg + tf.transpose(row_plus_dcg, perm=[1, 0])
        sub_dcg = row_sub_dcg + tf.transpose(row_sub_dcg, perm=[1, 0])
        ndcg_delta = tf.divide(plus_dcg - sub_dcg, idcg)
        
        # BPR Loss加权
        label_mat = tf.cast(tf.greater(true_sorted_by_pred,
                                       tf.transpose(true_sorted_by_pred, perm=[1, 0])), tf.float32)
        logits_mat = tf.sigmoid(tf.subtract(y_pred_sorted, tf.transpose(logits, perm=[1, 0])))
        loss_mat = label_mat * tf.math.log(logits_mat + eps) + \
                   (1.0 - label_mat) * tf.math.log(1.0 - logits_mat + eps)

        # NDCG@K的优化方法
        if self.k != None:
            topK_row_mask = tf.where(tf.cumsum(uid_mask, axis=0) <= self.k, 1.0, 0.0)
            topK_mask = topK_row_mask * tf.transpose(topK_row_mask, perm=[1, 0])
        else:
            topK_mask = tf.ones_like(loss_mat, tf.float32)

        loss_ = -tf.reduce_sum(tf.abs(ndcg_delta) * loss_mat * uid_mask * topK_mask)
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
    print(f"WeightedDistillRankingLoss:{DistillRankingLoss()(rels, preds, uid_tensor, use_dvalue=True)}")
    print(f"BucketDistillRankingLoss:{BucketDistillRankingLoss()(rels, preds, uid_tensor)}")
    print(f"LambdaLoss:{LambdaLoss(2)(rels, preds, uid_tensor)}")
