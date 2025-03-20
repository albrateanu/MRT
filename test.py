import argparse
import os
import cv2
import glob
import numpy as np
import math
import onnxruntime as ort

# -------------------------------
# Utility functions
# -------------------------------

def pad_to_factor(img, factor=8):
    """Pad image (H x W x C) so that its height and width are multiples of factor.
       Uses reflection padding.
    """
    H, W = img.shape[:2]
    new_H = math.ceil(H / factor) * factor
    new_W = math.ceil(W / factor) * factor
    pad_h = new_H - H
    pad_w = new_W - W
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    return padded, pad_h, pad_w

def unpad_image(img, pad_h, pad_w):
    """Remove bottom/right padding from image."""
    if pad_h == 0 and pad_w == 0:
        return img
    return img[:img.shape[0]-pad_h, :img.shape[1]-pad_w, :]

def run_onnx_inference(session, input_np):
    """Run inference using ONNX Runtime session.
       input_np: numpy array of shape (1, 3, H, W)
    """
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_np})
    return output[0]

def process_image(img, session, self_ensemble=False):
    """
    Process one image:
      - Convert from BGR to RGB and normalize to [0,1]
      - Pad to multiple of 8
      - If image dimensions are small (<3500x3500), run inference directly;
        otherwise, split along width (even and odd columns), infer separately, and merge.
      - Clamp the output, unpad, and convert back to uint8 BGR.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W, C = img.shape

    padded, pad_h, pad_w = pad_to_factor(img, factor=8)
    input_np = np.transpose(padded, (2, 0, 1))[np.newaxis, ...]

    if H < 3500 and W < 3500:
        out_np = run_onnx_inference(session, input_np)
    else:
        input_1 = input_np[:, :, :, 0::4]
        input_2 = input_np[:, :, :, 1::4]
        input_3 = input_np[:, :, :, 2::4]
        input_4 = input_np[:, :, :, 3::4]
        out1 = run_onnx_inference(session, input_1)
        out2 = run_onnx_inference(session, input_2)
        out3 = run_onnx_inference(session, input_3)
        out4 = run_onnx_inference(session, input_4)
        out_np = np.zeros_like(input_np)
        out_np[:, :, :, 0::4] = out1
        out_np[:, :, :, 1::4] = out2
        out_np[:, :, :, 2::4] = out3
        out_np[:, :, :, 3::4] = out4

    out_np = np.clip(out_np, 0, 1)
    out_img = np.transpose(out_np[0], (1, 2, 0))
    out_img = unpad_image(out_img, pad_h, pad_w)
    out_img = np.round(out_img * 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_img

# -------------------------------
# Main Inference Script
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference for Image Restoration")
    parser.add_argument("--comp", type=str, required=True, choices=["LLIE", "ShadowR"],
                        help="Competition type: LLIE or ShadowR")
    parser.add_argument("--self_ensemble", action="store_true",
                        help="(Optional) Use self-ensemble (currently not implemented for ONNX; flag ignored)")
    args = parser.parse_args()

    if args.comp == "LLIE":
        dataset_path = "./datasets/Test_Input"           # Path to LLIE images
        onnx_model_path = "./onnx_models/mrt_llie_80k.onnx"  # ONNX model for LLIE
        result_dir = "./results/LLIE"
    elif args.comp == "ShadowR":
        dataset_path = "./datasets/"          # Path to ShadowR images
        onnx_model_path = "./onnx_models/mrt_sr_120k.onnx"    # ONNX model for ShadowR
        result_dir = "./results/ShadowR"
    else:
        raise ValueError("Unknown competition type.")

    os.makedirs(result_dir, exist_ok=True)
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        print(f"✅ Loaded ONNX model from {onnx_model_path}")
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return

    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    if not image_paths:
        print(f"No images found in {dataset_path}")
        return
    print(f"Found {len(image_paths)} images in {dataset_path}.")

    for img_path in image_paths:
        print(f"Processing {img_path}...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read {img_path}. Skipping.")
            continue

        restored = process_image(img, session, self_ensemble=args.self_ensemble)
        
        rel_path = os.path.relpath(img_path, dataset_path)
        out_path = os.path.join(result_dir, rel_path)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save with maximum PNG compression
        cv2.imwrite(out_path, restored, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"Saved restored image to {out_path}")
        
    print("✅ Inference complete!")

if __name__ == "__main__":
    main()
