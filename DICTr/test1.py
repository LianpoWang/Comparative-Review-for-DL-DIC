import time
import torch
from thop import profile
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from utils.datasets import SpeckleDataset
from networks.dictr import DICTr


def test_speckle(model,
                     model_path,
                     device,
                     attn_splits_list=False,
                     corr_radius_list=False,
                     prop_radius_list=False):
    # Load the pre-trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    test_dataset = SpeckleDataset("/home/dell/DATA/wh/DATASET/Test_annotations1.csv",
                                 '/home/dell/DATA/wh/DATASET/Test1',
                                 10800)
    print('Number of validation image pairs:', len(test_dataset))

    # Initialize metrics
    mae_list = []
    mse_list =[]
    epe_list = []
    results = {}

    s00_05_list = []
    s05_10_list = []
    s10plus_list = []
    start_time=time.time()
    with torch.no_grad():
        for val_id in tqdm(range(len(test_dataset)), desc="Processing test set"):
            image1, image2, flow_gt, valid_gt = test_dataset[val_id]
            image1 = image1[None].to(device)
            image2 = image2[None].to(device)

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list)

            flow_pr = results_dict['flow_preds'][-1]
            flow = flow_pr[0].cpu()

            # Calculate errors
            error = flow - flow_gt
            abs_error = torch.abs(error)
            squared_error = error ** 2

            # MAE (Mean Absolute Error)
            mae = abs_error.mean(dim=0)

            # MSE (Mean Squared Error)
            mse = squared_error.mean(dim=0)

            # AEE (Average Endpoint Error)
            epe = torch.sum(squared_error, dim=0).sqrt()

            # Magnitude of ground truth flow
            mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
            flow_gt_speed = mag

            # Store metrics
            valid_mask = valid_gt >= 0.5
            if valid_mask.max() > 0:
                mae_list.append(mae[valid_mask].numpy())
                mse_list.append(mse[valid_mask].numpy())
                epe_list.append(epe[valid_mask].numpy())

            # Speed-based analysis
            valid_mask = (flow_gt_speed < 0.5)
            if valid_mask.max() > 0:
                s00_05_list.append(epe[valid_mask].numpy())

            valid_mask = (flow_gt_speed >= 0.5) * (flow_gt_speed <= 1)
            if valid_mask.max() > 0:
                s05_10_list.append(epe[valid_mask].numpy())

            valid_mask = (flow_gt_speed > 1)
            if valid_mask.max() > 0:
                s10plus_list.append(epe[valid_mask].numpy())
    total_time = (time.time() - start_time)*1000
    avg_time=total_time/len(test_dataset)
    print('Avg Inference time: {:.2f} ms'.format(avg_time))
    # Calculate final metrics
    mae_all = np.concatenate(mae_list)
    mse_all = np.concatenate(mse_list)
    epe_all = np.concatenate(epe_list)

    results['MAE'] = np.mean(mae_all)
    results['MSE'] = np.mean(mse_all)
    results['AEE'] = np.mean(epe_all)

    # Speed-based results
    results['AEE_s0_0.5'] = np.mean(np.concatenate(s00_05_list))
    results['AEE_s0.5_1'] = np.mean(np.concatenate(s05_10_list))
    results['AEE_s1+'] = np.mean(np.concatenate(s10plus_list))

    print("\nFinal Metrics:")
    print("MAE: %.4f" % results['MAE'])
    print("MSE: %.4f" % results['MSE'])
    print("AEE: %.4f" % results['AEE'])
    print("AEE by speed range:")
    print("  s0-0.5: %.4f" % results['AEE_s0_0.5'])
    print("  s0.5-1: %.4f" % results['AEE_s0.5_1'])
    print("  s1+: %.4f" % results['AEE_s1+'])
    flops, params = profile(model, inputs=(image1, image2), verbose=False)
    print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model with default parameters
    model = DICTr(feature_channels=128,
                  num_scales=2,
                  upsample_factor=2,
                  num_head=1,
                  attention_type='swin',
                  ffn_dim_expansion=4,
                  num_transformer_layers=12).to(device)

    # Path to your pre-trained model
    model_path = "" # Update this path

    # Validation parameters
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]

    # Run validation
    results = test_speckle(model,
                               model_path,
                               device,
                               attn_splits_list=attn_splits_list,
                               corr_radius_list=corr_radius_list,
                               prop_radius_list=prop_radius_list)

    # Save results to file
    output_file = "/home/dell/DATA/wh/dictr-main/result/test1_results.txt"
    with open(output_file, 'w') as f:
        f.write("Test Results:\n")
        f.write("MAE: %.4f\n" % results['MAE'])
        f.write("MSE: %.4f\n" % results['MSE'])
        f.write("AEE: %.4f\n" % results['AEE'])
        f.write("\nAEE by speed range:\n")
        f.write("s0-0.5: %.4f\n" % results['AEE_s0_0.5'])
        f.write("s0.5-1: %.4f\n" % results['AEE_s0.5_1'])
        f.write("s1+: %.4f\n" % results['AEE_s1+'])

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()