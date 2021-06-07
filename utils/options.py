import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=64, help="rounds of training")
    parser.add_argument('--factor', type=int, default=1, help="rounds of training")
    parser.add_argument('--ep_lr_decay', type=int, default=3, help="rounds of training")
    parser.add_argument('--cuda_device', type=str, default="6", help="cuda_device")
    parser.add_argument('--imshow', type=bool, default=False, help="show some training dataset")
    parser.add_argument('--model', type=str, default='stn_trans_shuf', help='model name')
    parser.add_argument('--train_data', type=str, default='PKLot', help='model name')
    parser.add_argument('--path', type=str, default="", help='trained model path')
    parser.add_argument('--train_img', type=str, default='/home/zengweijia/.jupyter/cnrpark/PKLot/PKLotSegmented',
                        help="path to training set images")
    parser.add_argument('--train_lab', type=str, default='/home/zengweijia/.jupyter/cnrpark/splits/PKLot/train_36.txt',
                        help="path to training set labels")
    parser.add_argument('--test_img', type=str, default='/home/zengweijia/.jupyter/cnrpark/PATCHES',
                        help="path to test set images")
    parser.add_argument('--test_lab', type=str, default='/home/zengweijia/.jupyter/cnrpark/splits/CNRPark-EXT/all.txt',
                        help="path to test set labels")
    # parser.add_argument('--train_img', type=str, default='/Users/julyc/Downloads/CNR-EXT-Patches-150x150/PATCHES',
    #                     help="path to training set images")
    # parser.add_argument('--train_lab', type=str, default='/Users/julyc/Downloads/splits/CNRPark-EXT/train.txt',
    #                     help="path to training set labels")
    # parser.add_argument('--test_img', type=str, default='/Users/julyc/Downloads/CNR-EXT-Patches-150x150/PATCHES',
    #                     help="path to test set images")
    # parser.add_argument('--test_lab', type=str, default='/Users/julyc/Downloads/splits/CNRPark-EXT/test.txt',
    #                     help="path to test set labels")
    parser.add_argument('--img_size', type=int, default=56,
                        help="carnet :54  malexnet=224")
    parser.add_argument('--split_path', type=str, default='/home/zengweijia/.jupyter/cnrpark/parking_lot_occupancy_detection/data/stn_trans_shuf',
                        help="path to training set labels")
    parser.add_argument('--use_mul_gpu', type=int, default=1, help="use mul-gpu or not")
    parser.add_argument('--eval_data', type=str,default='cnrext',help="path to training set labels")
    parser.add_argument('--use_transformer', default=True, type=bool)

    args = parser.parse_args()
    return args
