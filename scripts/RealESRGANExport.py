import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet


def main(args):
    # An instance of the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=args.scale)
    if args.scale == 8:
        model.load_state_dict(torch.load(args.input), strict=False)
    else:
        keyname = 'params_ema'
        model.load_state_dict(torch.load(args.input)[keyname], strict=False)
    
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, 64, 64)
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, args.output, opset_version=11, export_params=True)
    print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='models/pytorch/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan-x4.onnx', help='Output onnx path')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')
    #parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)