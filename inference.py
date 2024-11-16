"""
CUDA_VISIBLE_DEVCIES=0 python3 inference.py \
    -m swin_large_patch4_window7_224 \
    -c ./checkpoint-43.pth.tar \
    -b 16  \
    -td /data/datasets/cse_244/DeepLearning_Data/test/
"""

import argparse
import os
import timm
import torch
import pandas as pd

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for timm")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--ckpt", "-c", type=str, required=True,
                        help="Path to the checkpoint file")
    parser.add_argument("--output_dir", "-od", type=str, default="inference_results",
                        help="Directory to the inference results")
    parser.add_argument("--output_csv", "-oc", type=str, default="output.csv",
                        help="Name of the output CSV file")
    parser.add_argument("--num_classes", "-nc", type=int, default=100,
                        help="Number of the classes")
    parser.add_argument("--in_chans", "-in", type=int, default=3,
                        help="Number of the input channels")
    parser.add_argument("--batch", "-b", type=int, default=2,
                        help="Inference batch size")

    # Dataset properties
    parser.add_argument("--test_dir", "-td", type=str, required=True,
                        help="Path to the test dataset directory")
    parser.add_argument("--workers", "-w", type=int, default=16,
                        help="Number of workers")
    parser.add_argument("--topk", "-tk", type=int, default=1,
                        help="TopK value")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        pretrained=False,
        checkpoint_path=args.ckpt
    ).to(device)

    data_config = timm.data.resolve_data_config(model=args.model)
    dataset = timm.data.create_dataset(
        root=args.test_dir,
        name='',
        split='validation'
    )

    data_config = timm.data.resolve_data_config(model=args.model)
    loader = timm.data.create_loader(
        dataset,
        batch_size=args.batch,
        use_prefetcher=True,
        num_workers=args.workers,
        **data_config
    )
    all_indices = []
    all_outputs = []
    with torch.no_grad():
        for batch_idx, (inp, _) in enumerate(loader):
            print(f"Processed batch: {batch_idx}")
            inp = inp.to(device)
            output = model(inp)
            output = output.softmax(-1)
            output, indices = output.topk(args.topk)
            np_indices = indices.cpu().numpy()
            all_indices.append(np_indices)
            all_outputs.append(output.cpu().numpy())

    all_indices = np.concatenate(all_indices, axis=0).squeeze(-1)
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)

    data = {}
    data["ID"] = loader.dataset.filenames(basename=True)
    data["Label"] = all_indices

    data = pd.DataFrame(data=data)
    output_file_location = os.path.join(args.output_dir, args.output_csv)
    data.to_csv(output_file_location, index=False)
    print(f"Saved the results file at: {output_file_location}")


if __name__ == "__main__":
    main(parse_args())