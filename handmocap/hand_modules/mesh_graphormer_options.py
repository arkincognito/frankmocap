# Copyright (c) Facebook, Inc. and its affiliates.

# Part of the code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from .base_options import BaseOptions

class MeshGraphormerOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='-1', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--visualize_eval', action='store_true')
        self.parser.add_argument('--test_dataset', type=str, choices=['freihand', 'ho3d', 'stb', 'rhd', 'mtc', 'wild', 'demo'], help="which dataset to test on")
        self.parser.add_argument("--checkpoint_path", type=str, default=None, help="path of checkpoints used in test")
        self.isTrain = False
        #########################################################
        # Data related arguments
        #########################################################
        self.parser.add_argument("--num_workers", default=4, type=int, 
                            help="Workers in dataloader.")       
        self.parser.add_argument("--img_scale_factor", default=1, type=int, 
                            help="adjust image resolution.")  
        self.parser.add_argument("--image_file_or_path", default='./samples/hand', type=str, 
                            help="test data")
        #########################################################
        # Loading/saving checkpoints
        #########################################################
        self.parser.add_argument("--model_name_or_path", default='./handmocap/hand_modules/MeshGraphormer/src/modeling/bert/bert-base-uncased/', type=str, required=False,
                            help="Path to pre-trained transformer model or model type.")
        self.parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                            help="Path to specific checkpoint for resume training.")
        self.parser.add_argument("--output_dir", default='output/', type=str, required=False,
                            help="The output directory to save checkpoint and test results.")
        self.parser.add_argument("--config_name", default="", type=str, 
                            help="Pretrained config name or path if not the same as model_name.")
        self.parser.add_argument('-a', '--arch', default='hrnet-w64',
                        help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
        #########################################################
        # Model architectures
        #########################################################
        self.parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                            help="Update model config if given")
        self.parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                            help="Update model config if given")
        self.parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                            help="Update model config if given. Note that the division of "
                            "hidden_size / num_attention_heads should be in integer.")
        self.parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                            help="Update model config if given.")
        self.parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                            help="The Image Feature Dimension.")          
        self.parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                            help="The Image Feature Dimension.")  
        self.parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                            help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
        self.parser.add_argument("--mesh_type", default='hand', type=str, help="body or hand") 

        #########################################################
        # Others
        #########################################################
        self.parser.add_argument("--run_eval_only", default=True, action='store_true',) 
        self.parser.add_argument("--device", type=str, default='cuda', 
                            help="cuda or cpu")
        self.parser.add_argument('--seed', type=int, default=88, 
                            help="random seed for initialization.")