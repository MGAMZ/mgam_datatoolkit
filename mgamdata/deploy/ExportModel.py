'''
Jit Export nn.Module
'''

import torch

from mmseg.apis.inference import init_model
from mmseg.models.segmentors import EncoderDecoder


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Jit trace and export')
    parser.add_argument('cfg_path', type=str, help='Config file path',
                        default="/fileser51/zhangyiqin.sx/mmseg/work_dirs/0.8.3.FixRange/round_1/MedNext_3D/MedNext_3D.py")
    parser.add_argument('ckpt_path', type=str, help='Checkpoint file path',
                        default="/fileser51/zhangyiqin.sx/mmseg/work_dirs/0.8.3.FixRange/round_1/MedNext_3D/best_Perf_mDice_iter_24000.pth")
    parser.add_argument('save_path', type=str, help='Output path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model:EncoderDecoder = init_model(args.cfg_path, args.ckpt_path)
    model.requires_grad_(False)
    model.eval()
    exported = torch.jit.trace(
        func = model.whole_inference, 
        example_inputs = (torch.ones(1,1,512,512, 
                                        dtype=torch.float32, 
                                        device='cuda'))
    )
    output=exported(torch.ones(1,3,300,300).cuda())
    print(output.shape)
    # exported.save(args.save_path)