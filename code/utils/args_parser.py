import argparse
import sys

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def get_args():
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # dataset
    parser.add_argument('--data_dir', type=str, default='./')
    # parser.add_argument('--train_annos_file', type=str, default=None)
    # parser.add_argument('--val_annos_file', type=str, default=None)
    # parser.add_argument('--test_annos_file', type=str, default=None)
    parser.add_argument('--stage', default='ava', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['ava'], type=str, nargs='+',
                        help='validation datasets')

    # training parameters
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train, 128 for NIMA, 48 for TANet')
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='worker num of dataloader')
    parser.add_argument('--epochs', type=int, default=100, help='epoch num of train')
    # for NIMA
    parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    parser.add_argument('--dense_lr', type=float, default=5e-4)
    # for TANet
    parser.add_argument('--init_lr_res365_last', type=int, default=0.0000005, help='learning_rate')
    parser.add_argument('--init_lr_mobileNet', type=int, default=0.0000005, help='learning_rate')
    parser.add_argument('--init_lr_head', type=int, default=0.0000005, help='learning_rate')
    parser.add_argument('--init_lr_head_rgb', type=int, default=0.0000005, help='learning_rate')
    parser.add_argument('--init_lr_hypernet', type=int, default=0.0000005, help='learning_rate')
    parser.add_argument('--init_lr_tygertnet', type=int, default=0.0000005, help='learning_rate')
    parser.add_argument('--init_lr', type=int, default=0.0000005, help='learning_rate')


    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--use_gpus', default=None, type=str, help='GPUs to use, e.g. 0,1,2,3')
    # parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')
    # parser.add_argument('--gpu_ids', type=list, nargs='+', default=None)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./ckpts/tmp')
    parser.add_argument('--log_dir', type=str, default='./logs/tmp')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')

    
    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # model
    parser.add_argument('--model', type=str, default='nima')

    # parse args
    if sys.argv.__len__() == 2:
        # parse from file
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        # parse direct
        args = parser.parse_args()

    # check annotations json file
    if args.train:
        assert (args.stage == 'ava' or args.stage == 'yfcc' or args.stage == 'yfcc_clean'), 'train stage must be ava or yfcc or yfcc_clean'
    if args.stage == 'ava':
        assert ('ava' in args.val_dataset), 'ava must in val_dataset'
    elif args.stage == 'yfcc':
        assert ('yfcc1m' in args.val_dataset), 'yfcc1m must in val_dataset'
    elif args.stage == 'yfcc_clean':
        assert ('yfcc1m_clean' in args.val_dataset), 'yfcc1m_clean must in val_dataset'

    return args