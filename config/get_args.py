import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # =================== Training settings =======================
    parser.add_argument("--nepochs", type=int, default=100) # number of epochs
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=1) # batch size
    parser.add_argument("--nworkers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3) # learning rate
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    # nuscenes
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # dirs
    parser.add_argument("--savedir", type=str, default='./runs') # save dir
    parser.add_argument('--dataroot', type=str, default='../dataset') # root to nuscenes
    parser.add_argument('--seg_checkpoint_dir',default='',type=str) # checkpoint dir of BEV seg net
    parser.add_argument('--centerlinedet_checkpoint_dir',default='', type=str) # checkpoint dir of centerlinedet
    
    # DDP
    parser.add_argument('--local_rank', default=0, type=int)

    # half length of the cropping window size (only for centerlineder)
    parser.add_argument('--ROI_half_length',default=50,type=int)

    # training mode BC/DAgger
    parser.add_argument('--training_mode',default='BC',type=str)

    # max tolerated distance for DAgger sampling
    parser.add_argument('--tolerate_distance',default=10,type=int)

    # BC/DAgger sample rate (if rate=3, sampling every 3 frames for better efficiency)
    parser.add_argument('--sample_rate',default=3,type=int)
    
    # thresholds
    parser.add_argument('--binary_thr',default=0.5,type=float)# thr for binarize candidate initial vertex segmentation map
    parser.add_argument('--area_thr',default=4,type=int)# thr for filtering binalized candidate initial vertex area
    parser.add_argument('--logit_thr',default=0.9,type=float)# thr for extraction of valid vertices in the next step
    parser.add_argument('--filter_distance',default=15,type=int)# thr for filtering candidate initial vertices that are too closed to past trajectories
    parser.add_argument('--world_alignment_distance',default=1,type=float)# thr for aligning one instance endpoint to another if they are too closed
                                                                        # the distance here is in world coordinate, which is equal to 4 pixels in the local frame coordinate

    # sampler config
    parser.add_argument('--noise',default=3,type=int) # noise for BC sampling
    parser.add_argument('--step_size',default=20,type=int) # move step size for BC sampling

    # =================== BEV segmentment network ==================
    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # trainig set selection
    parser.add_argument('--train_set_selection', default='all', type=str,
                        help='training set selection')

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-25.0, 25.0, 0.25])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-25.0, 25.0, 0.25])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    
    
    # ===================== CenterLineDet transformer network ========================
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=False,
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    args = parser.parse_args()
    return args