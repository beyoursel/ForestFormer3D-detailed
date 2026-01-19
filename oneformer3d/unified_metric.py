import torch
import numpy as np
from scipy import stats
from mmengine.logging import MMLogger
from tqdm import tqdm

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval


@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """
        Compute metrics for online evaluation, similar to ForAINetV2 compute_metrics.

        Args:
            results (list): list of tuples (eval_ann, single_pred_results)
                eval_ann: dict with 'pts_semantic_mask' and 'pts_instance_mask'
                single_pred_results: dict with 'pts_semantic_mask' and 'pts_instance_mask'

        Returns:
            metrics (dict): dictionary containing semantic mIoU, binary mIoU, instance PQ/SQ/RQ, F1, MUCov, MWCov
        """
        logger: MMLogger = MMLogger.get_current_instance()

        #initialization
        NUM_CLASSES = 3  # @Treeins: classes unclassified, non-tree and tree
        NUM_CLASSES_SEM = 4
        # class index for instance segmenatation
        ins_classcount = [2]  # @Treeins
        # class index for stuff segmentation
        stuff_classcount = [1]  # @Treeins
        # class index for semantic segmenatation
        sem_classcount = [1, 2, 3] # @Treeins
        sem_classcount_have = []
        stuff_classes = [1]
        thing_classes = [2,3]

        true_positive_classes_global = np.zeros(NUM_CLASSES_SEM) # TP
        positive_classes_global = np.zeros(NUM_CLASSES_SEM)
        gt_classes_global = np.zeros(NUM_CLASSES_SEM) # global GT

        total_gt_ins_global = np.zeros(NUM_CLASSES) # total GT instances
        tpsins_global = [[] for _ in range(NUM_CLASSES)] # TP instances
        fpsins_global = [[] for _ in range(NUM_CLASSES)] # FP instances
        IoU_Tp_global = np.zeros(NUM_CLASSES) # TP IoU
        IoU_Mc_global = np.zeros(NUM_CLASSES) # MC IoU ???

        all_mean_cov_global = [[] for _ in range(NUM_CLASSES)]
        all_mean_weighted_cov_global = [[] for _ in range(NUM_CLASSES)]

        for idx, eval_data in enumerate(tqdm(results, desc="Evaluating...")):
            eval_ann = eval_data[0]
            single_pred_results = eval_data[1]
            # 处理单个场景的点云数据
            true_positive_classes = np.zeros(NUM_CLASSES_SEM)
            positive_classes = np.zeros(NUM_CLASSES_SEM)
            gt_classes = np.zeros(NUM_CLASSES_SEM)

            total_gt_ins = np.zeros(NUM_CLASSES)
            tpsins = [[] for _ in range(NUM_CLASSES)]
            fpsins = [[] for _ in range(NUM_CLASSES)]
            IoU_Tp = np.zeros(NUM_CLASSES)
            IoU_Mc = np.zeros(NUM_CLASSES)

            all_mean_cov = [[] for _ in range(NUM_CLASSES)]
            all_mean_weighted_cov = [[] for _ in range(NUM_CLASSES)]

            sem_pre_i = single_pred_results['pts_semantic_mask'] + 1 # change to [1, 2, 3]
            sem_gt_i = eval_ann['pts_semantic_mask'] + 1 # 真值label

            ins_pre_i_ori = single_pred_results['pts_instance_mask']
            ins_gt_i_ori = eval_ann['pts_instance_mask']

            pred_sem_complete = sem_pre_i
            gt_sem_complete = sem_gt_i
            pred_ins_complete = ins_pre_i_ori
            gt_ins_complete = ins_gt_i_ori

            idxc = ((gt_sem_complete != 0) & (gt_sem_complete != 1)) | ((pred_sem_complete != 0) & (pred_sem_complete != 1)) 
            pred_ins = pred_ins_complete[idxc] # 索引点有效点
            gt_ins = gt_ins_complete[idxc]
            pred_sem = pred_sem_complete[idxc]
            gt_sem = gt_sem_complete[idxc]
            # 对原始未进行筛选的pred_sem_complete进行统计
            for j in range(gt_sem_complete.shape[0]): # eval gt_sem 
                gt_l = int(gt_sem_complete[j])
                pred_l = int(pred_sem_complete[j])
                gt_classes[gt_l] += 1
                positive_classes[pred_l] += 1
                true_positive_classes[gt_l] += int(gt_l == pred_l) # TP

            predicted_labels_copy = pred_sem_complete.copy()
            for i in stuff_classes:
                pred_sem_complete[predicted_labels_copy == i] = 1
            for i in thing_classes:
                pred_sem_complete[predicted_labels_copy == i] = 2

            gt_labels_copy = gt_sem_complete.copy()
            for i in stuff_classes:
                gt_sem_complete[gt_labels_copy == i] = 1
            for i in thing_classes:
                gt_sem_complete[gt_labels_copy == i] = 2
            # 统计地面和树两个类别的结果
            true_positive_classes_bi = np.zeros(NUM_CLASSES)
            positive_classes_bi = np.zeros(NUM_CLASSES)
            gt_classes_bi = np.zeros(NUM_CLASSES)
            for j in range(gt_sem_complete.shape[0]):
                gt_l = int(gt_sem_complete[j])
                pred_l = int(pred_sem_complete[j])
                gt_classes_bi[gt_l] += 1 # gt_classes_bi统计二分类中各个类别的gt数量
                positive_classes_bi[pred_l] += 1 # 统计网络预测的各个类别正样本数量
                true_positive_classes_bi[gt_l] += int(gt_l == pred_l) # 统计TP数量
            # 对经过筛选的pred_sem进行统计，仅评估ground和tree两个类别
            predicted_labels_copy = pred_sem.copy()
            for i in stuff_classes:
                pred_sem[predicted_labels_copy == i] = 1 # ground
            for i in thing_classes:
                pred_sem[predicted_labels_copy == i] = 2 # 合并wood和leave为tree

            gt_labels_copy = gt_sem.copy()
            for i in stuff_classes:
                gt_sem[gt_labels_copy == i] = 1
            for i in thing_classes:
                gt_sem[gt_labels_copy == i] = 2

            un = np.unique(pred_ins) # 
            pts_in_pred = [[] for _ in range(NUM_CLASSES)]
            for g in un: # 对预测的每个实例进行统计
                if g == -1:
                    continue
                tmp = (pred_ins == g)
                sem_seg_i = int(stats.mode(pred_sem[tmp])[0]) # 统计tmp中出现最多的语义标签作为实例的类别标签
                pts_in_pred[sem_seg_i] += [tmp] # 分组存储各个类别的实例点

            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES)]
            for g in un: # 统计实例的ground truth
                if g == -1:
                    continue
                tmp = (gt_ins == g)
                sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
                pts_in_gt[sem_seg_i] += [tmp]
            # 统计实例覆盖率cov
            for i_sem in range(NUM_CLASSES):
                sum_cov = 0
                mean_cov = 0
                mean_weighted_cov = 0
                num_gt_point = 0
                if not pts_in_gt[i_sem] or not pts_in_pred[i_sem]:
                    all_mean_cov[i_sem].append(0)
                    all_mean_weighted_cov[i_sem].append(0)
                    continue # 若对应类别的实例为空，则cov指标直接为0
                for ins_gt in pts_in_gt[i_sem]: # 统计gt和pred实例之间的iou
                    ovmax = 0.
                    num_ins_gt_point = np.sum(ins_gt) # 统计实例gt的点数量
                    num_gt_point += num_ins_gt_point
                    for ins_pred in pts_in_pred[i_sem]:
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou # 统计与gt预测实例之间最大的iou

                    sum_cov += ovmax
                    mean_weighted_cov += ovmax * num_ins_gt_point # 根据gt点数量进行加权

                if len(pts_in_gt[i_sem]) != 0: # 若对应类别的gt的实例不为空
                    mean_cov = sum_cov / len(pts_in_gt[i_sem])
                    all_mean_cov[i_sem].append(mean_cov)

                    mean_weighted_cov /= num_gt_point
                    all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

            for i_sem in range(NUM_CLASSES): # 统计各个类别的实例
                if not pts_in_pred[i_sem]:
                    continue
                IoU_Tp_per = 0
                IoU_Mc_per = 0
                tp = [0.] * len(pts_in_pred[i_sem])
                fp = [0.] * len(pts_in_pred[i_sem])
                if pts_in_gt[i_sem]:
                    total_gt_ins[i_sem] += len(pts_in_gt[i_sem]) # 统计真值实例点的总数
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.
                    if not pts_in_gt[i_sem]: # gt为空，pred不为空，则为fp
                        fp[ip] = 1
                        continue
                    for ins_gt in pts_in_gt[i_sem]:
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou

                    if ovmax > 0:
                        IoU_Mc_per += ovmax
                    if ovmax >= 0.5: # iou大于0.5才被认为是tp
                        tp[ip] = 1  # true
                        IoU_Tp_per += ovmax
                    else:
                        fp[ip] = 1  # false positive

                tpsins[i_sem] += tp
                fpsins[i_sem] += fp
                IoU_Tp[i_sem] += IoU_Tp_per
                IoU_Mc[i_sem] += IoU_Mc_per

            # semantic results
            iou_list = []
            sem_classcount_have = []
            for i in range(NUM_CLASSES_SEM):
                if gt_classes[i] > 0:
                    sem_classcount_have.append(i)
                    iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
                else:
                    iou = 0.0
                iou_list.append(iou)

            iou_list_bi = []
            sem_classcount_have_bi = []
            for i in range(NUM_CLASSES):
                if gt_classes_bi[i] > 0:
                    sem_classcount_have_bi.append(i)
                    iou = true_positive_classes_bi[i] / float(gt_classes_bi[i] + positive_classes_bi[i] - true_positive_classes_bi[i])
                else:
                    iou = 0.0
                iou_list_bi.append(iou)

            MUCov = np.zeros(NUM_CLASSES)
            MWCov = np.zeros(NUM_CLASSES)
            for i_sem in range(NUM_CLASSES):
                MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
                MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

            precision = np.zeros(NUM_CLASSES)
            recall = np.zeros(NUM_CLASSES)
            RQ = np.zeros(NUM_CLASSES)
            SQ = np.zeros(NUM_CLASSES)
            PQ = np.zeros(NUM_CLASSES)
            PQStar = np.zeros(NUM_CLASSES)

            for i_sem in ins_classcount:
                if not tpsins[i_sem] or not fpsins[i_sem]:
                    continue
                tp = np.asarray(tpsins[i_sem]).astype(float)
                fp = np.asarray(fpsins[i_sem]).astype(float)
                tp = np.sum(tp)
                fp = np.sum(fp)
                if total_gt_ins[i_sem] == 0:
                    rec = 0
                else:
                    rec = tp / total_gt_ins[i_sem] # recall
                if (tp + fp) == 0:
                    prec = 0
                else:
                    prec = tp / (tp + fp) # 精度
                precision[i_sem] = prec
                recall[i_sem] = rec
                if (prec + rec) == 0:
                    RQ[i_sem] = 0
                else:
                    RQ[i_sem] = 2 * prec * rec / (prec + rec) # RQ衡量分的准不准
                if tp == 0:
                    SQ[i_sem] = 0
                else:
                    SQ[i_sem] = IoU_Tp[i_sem] / tp # SQ衡量分的细不细
                PQ[i_sem] = SQ[i_sem] * RQ[i_sem] # PQ衡量全景分割质量
                PQStar[i_sem] = PQ[i_sem]

            for i_sem in stuff_classcount:
                if iou_list_bi[i_sem] >= 0.5:
                    RQ[i_sem] = 1
                    SQ[i_sem] = iou_list_bi[i_sem]
                else:
                    RQ[i_sem] = 0
                    SQ[i_sem] = 0
                PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
                PQStar[i_sem] = iou_list_bi[i_sem]

            true_positive_classes_global += true_positive_classes
            positive_classes_global += positive_classes
            gt_classes_global += gt_classes

            total_gt_ins_global += total_gt_ins
            for i in range(NUM_CLASSES):
                tpsins_global[i] += tpsins[i]
                fpsins_global[i] += fpsins[i]
                IoU_Tp_global[i] += IoU_Tp[i]
                IoU_Mc_global[i] += IoU_Mc[i]

            for i in range(NUM_CLASSES):
                all_mean_cov_global[i] += all_mean_cov[i]
                all_mean_weighted_cov_global[i] += all_mean_weighted_cov[i]
        
        metrics = dict()

        # 统计所有场景的语义分割结果
        iou_list_global = []
        sem_classcount_have_global = []
        for i in range(NUM_CLASSES_SEM):
            if gt_classes_global[i] > 0:
                sem_classcount_have_global.append(i)
                iou_global = true_positive_classes_global[i] / float(gt_classes_global[i] + positive_classes_global[i] - true_positive_classes_global[i])
            else:
                iou_global = 0.0
            iou_list_global.append(iou_global)

        set1_global = set(sem_classcount)
        set2_global = set(sem_classcount_have_global)
        set3_global = set1_global & set2_global
        sem_classcount_final_global = list(set3_global)

        metrics['iou_semantic'] = iou_list_global
        metrics['mIoU'] = 1. * sum(iou_list_global) / len(sem_classcount_final_global)

        iou_list_bi_global = []
        sem_classcount_have_bi_global = []
        for i in range(NUM_CLASSES):
            if gt_classes_bi[i] > 0:
                sem_classcount_have_bi_global.append(i)
                iou_bi_global = true_positive_classes_bi[i] / float(gt_classes_bi[i] + positive_classes_bi[i] - true_positive_classes_bi[i])
            else:
                iou_bi_global = 0.0
            iou_list_bi_global.append(iou_bi_global)

        sem_classcount_bi_global = [1, 2]
        set1_bi_global = set(sem_classcount_bi_global)
        set2_bi_global = set(sem_classcount_have_bi_global)
        set3_bi_global = set1_bi_global & set2_bi_global
        sem_classcount_final_bi_global = list(set3_bi_global)
        
        # 统计合并类别后，所有场景下的总二分类的指标
        metrics['mIoU_binary'] = 1. * sum(iou_list_bi_global) / len(sem_classcount_final_bi_global)
   
        # 统计全景分割相关指标
        MUCov_global = np.zeros(NUM_CLASSES)
        MWCov_global = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            MUCov_global[i_sem] = np.mean(all_mean_cov_global[i_sem])
            MWCov_global[i_sem] = np.mean(all_mean_weighted_cov_global[i_sem])

        precision_global = np.zeros(NUM_CLASSES)
        recall_global = np.zeros(NUM_CLASSES)
        RQ_global = np.zeros(NUM_CLASSES)
        SQ_global = np.zeros(NUM_CLASSES)
        PQ_global = np.zeros(NUM_CLASSES)
        PQStar_global = np.zeros(NUM_CLASSES)
        set1_ins_global = set(ins_classcount)
        set2_ins_global = set(sem_classcount_have_global)
        set3_ins_global = set1_ins_global & set2_ins_global
        ins_classcount_final_global = list(set3_ins_global)

        for i_sem in ins_classcount:
            if not tpsins_global[i_sem] or not fpsins_global[i_sem]:
                continue
            tp_global = np.asarray(tpsins_global[i_sem]).astype(float)
            fp_global = np.asarray(fpsins_global[i_sem]).astype(float)
            tp_global = np.sum(tp_global)
            fp_global = np.sum(fp_global)
            if total_gt_ins_global[i_sem] == 0:
                rec_global = 0
            else:
                rec_global = tp_global / total_gt_ins_global[i_sem]
            if (tp_global + fp_global) == 0:
                prec_global = 0
            else:
                prec_global = tp_global / (tp_global + fp_global)
            precision_global[i_sem] = prec_global
            recall_global[i_sem] = rec_global
            if (prec_global + rec_global) == 0:
                RQ_global[i_sem] = 0
            else:
                RQ_global[i_sem] = 2 * prec_global * rec_global / (prec_global + rec_global)
            if tp_global == 0:
                SQ_global[i_sem] = 0
            else:
                SQ_global[i_sem] = IoU_Tp_global[i_sem] / tp_global
            PQ_global[i_sem] = SQ_global[i_sem] * RQ_global[i_sem]
            PQStar_global[i_sem] = PQ_global[i_sem]

        for i_sem in stuff_classcount:
            if iou_list_bi_global[i_sem] >= 0.5:
                RQ_global[i_sem] = 1
                SQ_global[i_sem] = iou_list_bi_global[i_sem]
            else:
                RQ_global[i_sem] = 0
                SQ_global[i_sem] = 0
            PQ_global[i_sem] = SQ_global[i_sem] * RQ_global[i_sem]
            PQStar_global[i_sem] = iou_list_bi_global[i_sem]

        if np.mean(precision_global[ins_classcount_final_global]) + np.mean(recall_global[ins_classcount_final_global]) == 0:
            F1_score_global = 0.0
        else:
            F1_score_global = (2 * np.mean(precision_global[ins_classcount_final_global]) * np.mean(recall_global[ins_classcount_final_global])) / (
                        np.mean(precision_global[ins_classcount_final_global]) + np.mean(recall_global[ins_classcount_final_global]))

        metrics['mMWCov'] = np.mean(MWCov_global[ins_classcount_final_global])
        metrics['mMUCov'] = np.mean(MUCov_global[ins_classcount_final_global])
        metrics['mPrecision'] = np.mean(precision_global[ins_classcount_final_global])
        metrics['mRecall'] = np.mean(recall_global[ins_classcount_final_global])
        metrics['F1'] = F1_score_global
        metrics['mSQ'] = np.mean(SQ_global[sem_classcount_final_bi_global])
        metrics['mRQ'] = np.mean(RQ_global[sem_classcount_final_bi_global])
        metrics['mPQ'] = np.mean(PQ_global[sem_classcount_final_bi_global])
        
        log_str = 'Evaluation Results:\n'
        iou_semantic_str = ", ".join([f"{x:.4f}" for x in metrics['iou_semantic']])
        log_str += f"IoU_semantic (all): [{iou_semantic_str}], mIoU: {metrics['mIoU']:.4f}, mIoU_binary: {metrics['mIoU_binary']:.4f}\n"
        log_str += f"mPQ: {metrics['mPQ']:.4f}, mSQ: {metrics['mSQ']:.4f}, mRQ: {metrics['mRQ']:.4f}\n"
        log_str += f"mPrecision: {metrics['mPrecision']:.4f}, mRecall: {metrics['mRecall']:.4f}, F1: {metrics['F1']:.4f}\n"
        log_str += f"mMUCov: {metrics['mMUCov']:.4f}, mMWCov: {metrics['mMWCov']:.4f}"
        logger.info(log_str)

        return metrics
        
@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(
                single_pred_results['pts_instance_mask'])
            pred_instance_labels.append(single_pred_results['instance_labels'])
            pred_instance_scores.append(single_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict
