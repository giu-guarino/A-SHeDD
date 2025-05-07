import sys
import os
from scipy import io
from param import config
import A_SHeDD as ashedd
import argparse

if __name__ == '__main__':
    # Creating Argument Parser
    parser = argparse.ArgumentParser(prog='A-SHeDD Training code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='',
                                     epilog='')

    # Required Arguments
    required = parser.add_argument_group('required named arguments')
    required.add_argument("-d", "--dataset", type=str, required=True, help='Name of the Dataset')
    required.add_argument("-s", "--source", type=str, required=True, help='Name of Source Modality')

    # Optional Arguments
    optional = parser._action_groups.pop()
    optional.add_argument('-n_gpu', "--gpu_number", type=str, default=0,
                          help='Number of the GPU on which to run the algorithm.')
    optional.add_argument("-ns", "--nsamples", type=int, nargs="+", default=[5, 10, 25, 50],
                          help="List of numbers of labelled target samples")
    optional.add_argument("-np", "--nsplits", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="List of splits")
    optional.add_argument("-o", "--out_dir", type=str, default='Results',
                          help='The directory in which save the outcome.')
    optional.add_argument("-ds", "--ds_dir", type=str, default='Datasets',
                          help='The directory containing the datasets.')
    parser._action_groups.append(optional)

    # Arguments Parsing
    arguments = parser.parse_args()

    methods = arguments.method
    backbone = arguments.backbone
    gpu = arguments.gpu_number
    nsamples_list = arguments.nsamples
    nsplits_list = arguments.nsplits
    ds_dir = arguments.ds_dir

    # ---- Dataset Check ----
    if arguments.dataset not in config['ds']:
        raise ValueError(f"Dataset '{arguments.dataset}' not found! Available datasets: {config['ds']}")
    ds_idx = config['ds'].index(arguments.dataset)
    # os.path.is_file(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_data_filtered.npy"))

    # ---- Modality Check ----
    if arguments.source not in config['data_names'][arguments.dataset]:
        raise ValueError(
            f"Source modality '{arguments.source}' not found for dataset '{arguments.dataset}'. Available: {config['data_names'][arguments.dataset]}")
    source_idx = config['data_names'][arguments.dataset].index(arguments.source)

    out_dir = os.path.join(arguments.out_dir, config['ds'][ds_idx], 'A-SHeDD')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for nsamples in nsamples_list:
        table = []
        for nsplit in nsplits_list:
            print(
                f"Processing {arguments.dataset}. Source = {arguments.source}. nsplit = {nsplit}, nsamples = {nsamples}")
            sys.stdout.flush()

            #f1_val = train_fn[m](ds_dir, out_dir, nsamples, nsplit, ds_idx, source_idx, gpu, backbone)
            f1_val, f1_nw = ashedd.train_fn(ds_dir, out_dir, nsamples, nsplit, ds_idx, source_idx, gpu)

            #table.append([ds_idx, source_idx, nsplit, nsamples, f1_val])  # Save results
            table.append([ds_idx, source_idx, nsplit, nsamples, f1_val, f1_nw])  # Save results

            io.savemat(
                os.path.join(out_dir, f'results_{arguments.dataset}_{arguments.source}_{nsamples}.mat'),
                {'RESULTS': table}
            )

'''
Example of istruction:
    python main.py -d EUROSAT-MS-SAR -s MS -n_gpu 0 -ds /home/giuseppe/Datasets
    
    python main.py -d RESISC45_EURO -s EURO -b CNN -m FLATMATCH SOFTMATCH -n_gpu 0 -ns 5 10 25 50 -np 1 2 3 4 5 -ds /home/giuseppe/Datasets
'''