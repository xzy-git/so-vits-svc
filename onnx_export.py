import argparse
import os
import json
import torch
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from fairseq import checkpoint_utils
from onnxexport.model_onnx import SynthesizerTrn
import utils

def get_hubert_model():
    vec_path = "hubert/checkpoint_best_legacy_500.pt"
    print("load model(s) from {}".format(vec_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [vec_path],
        suffix="",
    )
    model = models[0]
    model.eval()
    return model


def main(NetExport,Netpath,Netfile,Projectname):
    '''if HubertExport:
        device = torch.device("cpu")
        vec_path = "hubert/checkpoint_best_legacy_500.pt"
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [vec_path],
            suffix="",
        )
        original = models[0]
        original.eval()
        model = original
        test_input = torch.rand(1, 1, 16000)
        model(test_input)
        torch.onnx.export(model,
                          test_input,
                          "hubert4.0.onnx",
                          export_params=True,
                          opset_version=16,
                          do_constant_folding=True,
                          input_names=['source'],
                          output_names=['embed'],
                          dynamic_axes={
                              'source':
                                  {
                                      2: "sample_length"
                                  },
                          }
                          )'''

    if NetExport:
        device = torch.device("cpu")
        hps = utils.get_hparams_from_file(f"{Netpath}/config.json")
        SVCVITS = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        _ = utils.load_checkpoint(f"{Netpath}/{Netfile}.pth", SVCVITS, None)
        _ = SVCVITS.eval().to(device)
        for i in SVCVITS.parameters():
            i.requires_grad = False
        test_hidden_unit = torch.rand(1, 10, 256)
        test_pitch = torch.rand(1, 10)
        test_mel2ph = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
        test_uv = torch.ones(1, 10, dtype=torch.float32)
        test_noise = torch.randn(1, 192, 10)
        test_sid = torch.LongTensor([0])
        input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
        output_names = ["audio", ]
        SVCVITS.eval()

        onnx_path=f"{Netpath}/{Projectname}-{Netfile}"
        if not os.path.exists(onnx_path):
            os.makedirs(onnx_path)

        torch.onnx.export(SVCVITS,
                          (
                              test_hidden_unit.to(device),
                              test_pitch.to(device),
                              test_mel2ph.to(device),
                              test_uv.to(device),
                              test_noise.to(device),
                              test_sid.to(device)
                          ),
                          f"{onnx_path}/{Projectname}-{Netfile}_SoVits.onnx",
                          dynamic_axes={
                              "c": [0, 1],
                              "f0": [1],
                              "mel2ph": [1],
                              "uv": [1],
                              "noise": [2],
                          },
                          do_constant_folding=False,
                          opset_version=16,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names)

        with open(f"{Netpath}/config.json", 'r') as f:
            train_config=json.load(f)

        Rate=train_config['data']['sampling_rate']
        Hop=train_config['data']['hop_length']
        Characters=list(train_config['spk'].keys())

        infer_config=\
        {\
           "Folder" : "NyaruTaffySo",\
            "Name" : "NyaruTaffy-SoVits",\
            "Type" : "SoVits",\
            "Rate" : 44100,\
            "Hop" : 512,\
            "Cleaner" : "",\
            "Hubert": "hubert4.0",\
            "SoVits4": True,\
            "Characters" : ["Taffy","Nyaru"]\
        }

        infer_config['Folder']=f"{Projectname}-{Netfile}"
        infer_config['Name']=f"{Projectname}-{Netfile}"
        infer_config['Rate']=Rate
        infer_config['Hop']=Hop
        infer_config['Characters']=Characters

        with open(f"{Netpath}/{Projectname}-{Netfile}.json", 'w') as f:
            json.dump(infer_config,f)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export_model',type=str,help='Export the model or not')
    parser.add_argument('--model_path',type=str,help='The path of the model')
    parser.add_argument('--model_file',type=str,help='The filename of the model')
    parser.add_argument('--project_name',type=str,help='Project name, affect the name of the model folder, model file and config file')

    args = parser.parse_args()
    if not args.export_model or not args.model_path or not args.model_file or not args.project_name:
        parser.print_help()
        sys.exit(0)
    else:
        print(args.export_model)
        print(args.model_path)
        print(args.model_file)
        print(args.project_name)

        export_model=True
        if(args.export_model=='True'):
            export_model=True
        elif(args.export_model=='False'):
            export_model=False

        main(export_model,args.model_path,args.model_file,args.project_name)