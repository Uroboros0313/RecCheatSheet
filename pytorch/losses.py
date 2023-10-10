import torch


def dvalue_weighted_distill_ranking_loss(logits, y_true, uid_tensor, eps=1e-4):
    '''
    
    '''
    assert logits.shape == y_true.shape
    assert logits.shape == uid_tensor.shape
    
    uid_mask = (uid_tensor == uid_tensor.transpose(0, 1)).float()
    label_mat = (y_true > y_true.transpose(0, 1))
    dvalue_mat = (y_true - y_true.transpose(0, 1))
    logits_mat = (logits - logits.transpose(0, 1)).float()
    
    score = torch.sigmoid(logits_mat)
    loss_mat = torch.log(score + eps) * label_mat
    loss_ = - torch.sum(uid_mask * loss_mat * dvalue_mat)
    
    return loss_


def distill_ranking_loss(logits, y_true, uid_tensor, eps=1e-4):
    '''
    
    '''
    assert logits.shape == y_true.shape
    assert logits.shape == uid_tensor.shape
    
    uid_mask = (uid_tensor == uid_tensor.transpose(0, 1)).float()
    label_mat = (y_true > y_true.transpose(0, 1))
    logits_mat = (logits - logits.transpose(0, 1)).float()
    
    score = torch.sigmoid(logits_mat)
    loss_mat = torch.log(score + eps) * label_mat
    loss_ = - torch.sum(uid_mask * loss_mat)
    
    return loss_


def bpr_loss(logits, y_true, uid_tensor, eps=1e-4):
    '''
    
    '''
    assert logits.shape == y_true.shape
    assert logits.shape == uid_tensor.shape
    
    uid_mask = (uid_tensor == uid_tensor.transpose(0, 1)).float()
    label_mask = (y_true == y_true.transpose(0, 1)).float()
    label_mat = ((y_true - y_true.transpose(0, 1)) + 1) / 2
    logits_mat = (logits - logits.transpose(0, 1)).float()
    
    score = torch.sigmoid(logits_mat)
    loss_mat = torch.log(score + eps) * label_mat + torch.log(1 - score + eps) * (1 - label_mat)
    loss_ = - torch.sum(label_mask * uid_mask * loss_mat)
    
    return loss_

    
def listnet_loss(logits, y_true, uid_tensor, eps=1e-4):
    '''
    ListNet
    '''
    assert logits.shape == y_true.shape
    assert logits.shape == uid_tensor.shape
    
    uid_mask = (uid_tensor == uid_tensor.transpose(0, 1)).float()
    label_mask = torch.diag(y_true.squeeze())
    logits_mat = logits.repeat(1, logits.shape[0]).T
    
    pos_indices = (y_true.reshape(-1,) > 0.99)
    pos_score = torch.sum(torch.exp(uid_mask * logits_mat) * label_mask, dim=-1)
    sum_ = torch.sum(torch.exp(uid_mask * logits_mat), dim=-1)
    norm_socre = (pos_score / sum_)[pos_indices]
    
    loss_ = - torch.sum(torch.log(norm_socre + eps))
    
    return loss_


def listmle_loss(logits, y_true, uid_tensor, eps=1e-4):
    '''
    ListMLE
    '''
    assert logits.shape == y_true.shape
    assert logits.shape == uid_tensor.shape
    
    _, indices = torch.sort(y_true, descending=True, dim=0)
    indices = indices.squeeze()
    pred_sorted = logits[indices, :]
    uid_sorted = uid_tensor[indices, :]

    uid_mask = (uid_sorted == uid_sorted.transpose(0, 1)).float()
    pred_mat = torch.exp(pred_sorted.repeat(1, pred_sorted.shape[0]).T)
    cum_mask = torch.triu(torch.ones(pred_mat.shape))

    score = torch.diag(pred_mat)
    sum_ = torch.sum(pred_mat * uid_mask * cum_mask, dim=-1)
    norm_score = score / sum_

    loss_ = - torch.sum(torch.log(norm_score + eps))
    
    return loss_



if __name__ == '__main__':
    mock_uid = [1, 1, 1, 4, 4, 4, 4]
    mock_true = [0, 1, 1, 0, 0, 0, 1]
    mock_rel = [0.01, 0.9, 0.87, 0.12, 0.10, 0.13, 0.68]
    mock_pred = [0.5, 0.8, 0.2, 0.01, 0.23, 0.4, 0.1]


    y_pred = torch.tensor(mock_pred).reshape(-1, 1) * 2 - 1 # 模拟l2_norm后的结果
    y_true = torch.tensor(mock_true).reshape(-1, 1)
    y_rel = torch.tensor(mock_rel).reshape(-1, 1)
    uid_tensor = torch.tensor(mock_uid, dtype=torch.int32).reshape(-1, 1)
    
    print(listnet_loss(y_pred, y_true, uid_tensor))
    print(listmle_loss(y_pred, y_rel, uid_tensor))
    print(bpr_loss(y_pred, y_true, uid_tensor))
    print(dvalue_weighted_distill_ranking_loss(y_pred, y_rel, uid_tensor))
    print(distill_ranking_loss(y_pred, y_rel, uid_tensor))