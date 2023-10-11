import tensorflow as tf


class DistillRankingLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, logits, y_true, uid_tensor, use_dvalue=False, eps=1e-4):
        return self.__loss_fn__(logits, y_true, uid_tensor, use_dvalue, eps)

    def __loss_fn__(self, logits, y_true, uid_tensor, use_dvalue, eps):
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

    def __call__(self, logits, y_true, uid_tensor, use_dvalue=False, eps=1e-4):
        bucketized_y_true = self.__bucketize__(y_true)
        return self.__loss_fn__(logits, bucketized_y_true, uid_tensor, use_dvalue, eps)


class BprLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, logits, y_true, uid_tensor, eps=1e-4):
        return self.__loss_fn__(logits, y_true, uid_tensor, eps)

    def __loss_fn__(self, logits, y_true, uid_tensor, eps):
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

    def __call__(self, logits, y_true, uid_tensor=None, eps=1e-4):
        return self.__loss_fn__(logits, y_true, uid_tensor, eps)

    def __loss_fn__(self, logits, y_true, uid_tensor, eps):
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

    def __call__(self, logits, y_true, uid_tensor, eps=1e-4):
        return self.__loss_fn__(logits, y_true, uid_tensor, eps)

    def __loss_fn__(self, logits, y_true, uid_tensor, eps):
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


if __name__ == '__main__':
    uid_tensor = tf.reshape(tf.constant([1, 1, 1, 4, 4, 4, 4]), [-1, 1])
    true_ = tf.reshape(tf.cast([0, 1, 1, 0, 0, 0, 1], dtype=tf.float32), [-1, 1])
    rels = tf.reshape(tf.constant([0.01, 0.9, 0.87, 0.12, 0.10, 0.13, 0.68]), [-1, 1])
    preds = tf.reshape(tf.constant([0.5, 0.8, 0.2, 0.01, 0.23, 0.4, 0.1]), [-1, 1]) * 2 - 1

    print(f"ListMLELoss:{ListMLELoss()(preds, rels, uid_tensor)}")
    print(f"ListNetLoss:{ListNetLoss()(preds, true_, uid_tensor)}")
    print(f"BprLoss:{BprLoss()(preds, true_, uid_tensor)}")
    print(f"DistillRankingLoss:{DistillRankingLoss()(preds, rels, uid_tensor)}")
    print(f"WeightedDistillRankingLoss:{DistillRankingLoss()(preds, rels, uid_tensor, use_dvalue=True)}")
    print(f"BucketDistillRankingLoss:{BucketDistillRankingLoss()(preds, rels, uid_tensor)}")
