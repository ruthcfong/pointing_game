import numpy as np


def compute_metric(records, metric='pointing', idx=None):
    N, C = records.shape
    if idx is None:
        example_idx, class_idx = np.where(records != 0)
    else:
        example_idx, class_idx = np.where(idx)

    if metric == 'pointing':
        hits = np.zeros(C)
        misses = np.zeros(C)
    elif metric == 'average_precision':
        sum_precs = np.zeros(C)
        num_examples = np.zeros(C)
    else:
        assert(False)

    count = 0
    for i in range(len(example_idx)):
        j = example_idx[i]
        c = class_idx[i]
        rec = records[j, c]
        if metric == 'pointing':
            if rec == 1:
                hits[c] += 1
            elif rec == -1:
                misses[c] += 1
            else:
                count += 1
        elif metric == 'average_precision':
            sum_precs[c] += rec
            num_examples[c] += 1
        else:
            assert(False)
    print(count)

    if metric == 'pointing':
        acc = hits / (hits + misses)
        avg_acc = np.mean(acc)
        print('Avg Acc: %.4f' % avg_acc)
        for c in range(len(acc)):
            print(acc[c])
        return acc, avg_acc
    elif metric == 'average_precision':
        class_mean_avg_prec = sum_precs / num_examples
        mean_avg_prec = np.mean(class_mean_avg_prec)
        print('Mean Avg Prec: %.4f' % mean_avg_prec)
        for c in range(len(class_mean_avg_prec)):
            print(class_mean_avg_prec[c])
        return class_mean_avg_prec, mean_avg_prec
    else:
        assert(False)



def compute_metrics(out_path, metric='pointing', dataset='voc_2007'):
    records = np.loadtxt(out_path)
    hard_idx = np.loadtxt(f'data/hard_{dataset}.txt', delimiter=',')
    assert(records.shape == hard_idx.shape)
    print(f'Computing metrics from {out_path}')
    print(f'Overall Performance on {dataset}')
    compute_metric(records, metric=metric)
    print(f'Difficult Performance on {dataset}')
    compute_metric(records, metric=metric, idx=hard_idx)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    try:
        parser = argparse.ArgumentParser(description='Learn perturbation mask')
        parser.add_argument('--out_path', type=str, default=None)
        parser.add_argument('--dataset',
                            choices=['voc_2007', 'coco_2014', 'coco_2017'],
                            default='voc_2007',
                            help='name of dataset')
        parser.add_argument('--metric',
                            type=str,
                            choices=['pointing', 'average_precision'],
                            default='pointing')

        args = parser.parse_args()
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    compute_metrics(out_path=args.out_path,
                    dataset=args.dataset,
                    metric=args.metric)
