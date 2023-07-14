import numpy as np
import torch
import heapq
from EduNLP.Pretrain import BertTokenizer


def get_onehot_label_topk(classes_border_list, classes_offset_list, scores_list: np.ndarray, top_num=1):
    """
    Get the predicted labels based on the topK.

    Args:
        classes_border_list
        classes_offset_list
        scores_list: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    pred_onehot_labels = []
    scores_list = np.ndarray.tolist(scores_list)
    border, offset = classes_border_list, classes_offset_list
    num_level = len(border)
    for scores in scores_list:
        onehot_labels_list = [0] * len(scores)
        hlabels = {}
        for level in range(num_level):
            begin, end = border[level][0], border[level][1]
            cur_scores = scores[begin: end + 1]
            cur_offset = offset[level]
            cur_onehot_labels_list = [0] * len(cur_scores)
            # pred_onehot_scores[level].append(cur_scores)
            max_num_index_list = list(map(cur_scores.index, heapq.nlargest(top_num, cur_scores)))
            for i in max_num_index_list:
                cur_onehot_labels_list[i] = 1
                onehot_labels_list[i + cur_offset] = 1
            hlabels[level] = cur_onehot_labels_list
        # pred_onehot_scores[-1].append(scores)
        hlabels[num_level] = onehot_labels_list
        pred_onehot_labels.append(hlabels)
    return pred_onehot_labels


def compute_perfs(pred_labels: np.ndarray, true_labels: np.ndarray) -> tuple:
    # TP: number of labels which is predicted as True and is actually True.
    TP = np.sum(pred_labels * true_labels)
    # FP: number of labels which is predicted as True and is actually False.
    FP = np.sum(((pred_labels - true_labels) > 0).astype(np.int32))
    # FN: number of labels which is predicted as False and is actually True.
    FN = np.sum(((true_labels - pred_labels) > 0).astype(np.int32))
    # FP: number of labels which is predicted as False and is actually False.
    TN = np.sum(((pred_labels + true_labels) == 0).astype(np.int32))
    return np.array([TP, FP, FN, TN], dtype=np.int32)


def compute_perfs_per_layer(outputs: np.ndarray, true_labels: np.ndarray, hierarchy: dict, classes_border_list: list, keep_consistency: bool = True, threshold=0.5) -> tuple:
    def _make_labels_consistent(input_labels: np.ndarray, hierarchy: dict):
        input_labels = input_labels.astype(np.int32)
        while len(hierarchy) > 0:
            bottom_labels = set(hierarchy.keys()) - set(hierarchy.values())
            for child in bottom_labels:
                mask = (input_labels[:, child] == 1).astype(np.int32)
                input_labels[:, hierarchy[child]] |= mask
            for k in bottom_labels:
                hierarchy.pop(k)
        return input_labels

    preds = []
    for (start, end) in classes_border_list:
        threshold_labels = (outputs[:, start:end + 1] >= threshold).astype(np.int32)
        max_labels = (outputs[:, start:end + 1] == outputs[:, start:end + 1].max(axis=1)[:,None]).astype(np.int32)
        preds.append(threshold_labels | max_labels)
    pred_labels = np.concatenate(preds, axis=-1)
    del preds
    if keep_consistency:
        pred_labels = _make_labels_consistent(pred_labels, hierarchy.copy())
        true_labels = _make_labels_consistent(true_labels, hierarchy.copy())
    # get perfs per layer
    perfs_per_layer = []
    for (start, end) in classes_border_list:
        perfs_per_layer.append(compute_perfs(pred_labels[:, start:end + 1], true_labels[:, start:end + 1]))
    total_perfs = compute_perfs(pred_labels, true_labels)
    return perfs_per_layer, total_perfs


def compute_topk_recall(topk_preds: list, true_labels: list) -> tuple:
    rs = []
    for pred, label in zip(topk_preds, true_labels):
        _r = len(set(pred) & set(label)) / len(label)
        rs.append(_r)
    return np.mean(rs)


def quantile(array: torch.Tensor, ratio: float):
    """
    get quantile of array
    """
    assert ratio >= 0 and ratio <= 1
    assert len(array.shape) == 1
    sorted_array = torch.sort(array, dim=-1, descending=True)[0]
    index = min(int(len(array) * ratio + 0.5), len(array))
    return sorted_array[index].item()


def metric(TP, FP, FN, TN):
    def _f1_score(precision, recall):
        if precision + recall == 0:
            return 0.
        else:
            return 2 * precision * recall / (precision + recall)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    micro_f1 = _f1_score(precision, recall)
    acc = (TP + TN) / (TP + FP + FN + TN)
    return precision, recall, micro_f1, acc