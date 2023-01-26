import argparse
import time
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime as ort
import onnxoptimizer
import torch
from model_onnx import SynthesizerTrn
import utils
from hubert import hubert_model_onnx

def main(HubertExport,NetExport,Netpath,Netfile):

    if(HubertExport):
        device = torch.device("cuda")
        hubert_soft = hubert_model_onnx.hubert_soft("hubert/hubert-soft-0d54a1f4.pt")
        test_input = torch.rand(1, 1, 16000)
        input_names = ["source"]
        output_names = ["embed"]
        torch.onnx.export(hubert_soft.to(device),
                        test_input.to(device),
                        "hubert3.0.onnx",
                        dynamic_axes={
                            "source": {
                                2: "sample_length"
                            }
                        },
                        verbose=False,
                        opset_version=13,
                        input_names=input_names,
                        output_names=output_names)
    if(NetExport):
        device = torch.device("cuda")
        hps = utils.get_hparams_from_file(f"{Netpath}/config.json")
        SVCVITS = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        _ = utils.load_checkpoint(f"{Netpath}/{Netfile}.pth", SVCVITS, None)
        _ = SVCVITS.eval().to(device)
        for i in SVCVITS.parameters():
            i.requires_grad = False
        test_hidden_unit = torch.rand(1, 50, 256)
        test_lengths = torch.LongTensor([50])
        test_pitch = torch.rand(1, 50)
        test_sid = torch.LongTensor([0])
        input_names = ["hidden_unit", "lengths", "pitch", "sid"]
        output_names = ["audio", ]
        SVCVITS.eval()
        torch.onnx.export(SVCVITS,
                        (
                            test_hidden_unit.to(device),
                            test_lengths.to(device),
                            test_pitch.to(device),
                            test_sid.to(device)
                        ),
                        f"{Netpath}/{Netfile}.onnx",
                        dynamic_axes={
                            "hidden_unit": [0, 1],
                            "pitch": [1]
                        },
                        do_constant_folding=False,
                        opset_version=16,
                        verbose=False,
                        input_names=input_names,
                        output_names=output_names)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export_hubert',type=str,help='Export hubert or not')
    parser.add_argument('--export_model',type=str,help='Export the model or not')
    parser.add_argument('--model_path',type=str,help='The path of the model')
    parser.add_argument('--model_file',type=str,help='The filename of the model')

    args = parser.parse_args()
    if not args.export_hubert or not args.export_model or not args.model_path or not args.model_file:
        parser.print_help()
        sys.exit(0)
    else:
        print(args.export_hubert)
        print(args.export_model)
        print(args.model_path)
        print(args.model_file)

        export_hubert=True
        if(args.export_hubert=='True'):
            export_hubert=True
        elif(args.export_hubert=='False'):
            export_hubert=False

        export_model=True
        if(args.export_model=='True'):
            export_model=True
        elif(args.export_model=='False'):
            export_model=False

        main(export_hubert,export_model,args.model_path,args.model_file)