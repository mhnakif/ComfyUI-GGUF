# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import torch
import numpy as np
import gguf # This needs to be the llama.cpp one specifically!
import argparse
from tqdm import tqdm

from safetensors.torch import load_file

# Dictionary to map GGMLQuantizationType names to LlamaFileType names
quant_to_llama = {
    "F32": "ALL_F32",
    "F16": "MOSTLY_F16",
    "Q4_0": "MOSTLY_Q4_0",
    "Q4_1": "MOSTLY_Q4_1",
    "Q5_0": "MOSTLY_Q5_0",
    "Q5_1": "MOSTLY_Q5_1",
    "Q8_0": "MOSTLY_Q8_0",
    "Q8_1": "MOSTLY_Q8_1",
    "Q2_K": "MOSTLY_Q2_K_S",
    "Q3_K": "MOSTLY_Q3_K_S",
    "Q4_K": "MOSTLY_Q4_K_S",
    "Q5_K": "MOSTLY_Q5_K_S",
    "Q6_K": "MOSTLY_Q6_K_S",
    "Q8_K": "MOSTLY_Q8_K_S",
    "IQ2_XXS": "MOSTLY_IQ2_XXS",
    "IQ2_XS": "MOSTLY_IQ2_XS",
    "IQ3_XXS": "MOSTLY_IQ3_XXS",
    "IQ1_S": "MOSTLY_IQ1_S",
    "IQ4_NL": "MOSTLY_IQ4_NL",
    "IQ3_S": "MOSTLY_IQ3_S",
    "IQ2_S": "MOSTLY_IQ2_S",
    "IQ4_XS": "MOSTLY_IQ4_XS",
    "I8": "MOSTLY_IQ1_S",
    "I16": "MOSTLY_IQ1_M",
    "I32": "MOSTLY_IQ2_M",
    "I64": "MOSTLY_IQ3_M",
    "F64": "MOSTLY_F64",
    "IQ1_M": "MOSTLY_IQ1_M",
    "BF16": "MOSTLY_BF16",
    "Q4_0_4_4": "MOSTLY_Q4_0_4_4",
    "Q4_0_4_8": "MOSTLY_Q4_0_4_8",
    "Q4_0_8_8": "MOSTLY_Q4_0_8_8"
}

# You can now use this dictionary to map between the quantization types and Llama file types


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GGUF files from single SD ckpt")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output  unet gguf file.")
    parser.add_argument("--qtype", default="F16", help="Quant type [default: f16]")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    if args.dst is None:
        args.dst = os.path.splitext(args.src)[0] + f"_{args.qtype}.gguf"
        args.dst = os.path.basename(args.dst)

    if os.path.isfile(args.dst):
        input("Output exists enter to continue or ctrl+c to abort!")
    
    try:
        args.ftype = getattr(gguf.LlamaFileType, quant_to_llama[args.qtype])
        args.qtype = getattr(gguf.GGMLQuantizationType, args.qtype)
    except AttributeError:
        parser.error(f"Unknown quant/file type {args.qtype}")
    
    return args

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict)
    else:
        state_dict = load_file(path)
    return state_dict

def load_model(args):
    state_dict = load_state_dict(args.src)

    # from ComfyUI model detection
    if "transformer_blocks.0.attn.norm_added_k.weight" in state_dict:
        arch = "flux"
        raise ValueError(f"The Diffusers UNET can not be used for this!")
    elif "double_blocks.0.img_attn.proj.weight" in state_dict:
        arch = "flux" # mmdit ...?
    elif "transformer_blocks.0.attn.add_q_proj.weight" in state_dict:
        arch = "sd3"
    elif "down_blocks.0.downsamplers.0.conv.weight" in state_dict:
        if "add_embedding.linear_1.weight" in state_dict:
            arch = "sdxl"
        else:
            arch = "sd1" 
    else:
        breakpoint()
        raise ValueError(f"Unknown model architecture!")

    writer = gguf.GGUFWriter(path=None, arch=arch)
    return (writer, state_dict)

def handle_metadata(args, writer, state_dict):
    # TODO: actual metadata
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(args.ftype)

def handle_tensors(args, writer, state_dict):
    # TODO list:
    # - do something about this being awful and hacky

    max_name_len = max([len(s) for s in state_dict.keys()]) + 4
    for key, data in tqdm(state_dict.items()):
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32)
        data = data.numpy()

        old_dtype = data.dtype

        n_dims = len(data.shape)
        data_qtype = args.qtype
        data_shape = data.shape

        # get number of parameters (AKA elements) in this tensor
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        fallback = gguf.GGMLQuantizationType.F16

        # keys to keep as max precision
        blacklist = [
            "time_embedding.",
            "add_embedding.",
            "time_in.",
            "txt_in.",
            "vector_in.",
            "img_in.",
            "guidance_in.",
            "final_layer.",
        ]

        if any([x in key for x in blacklist]) and ".weight" in key:
            data_qtype = fallback

        if n_dims == 1: 
            # one-dimensional tensors should be kept in F32
            # also speeds up inference due to not dequantizing
            data_qtype = gguf.GGMLQuantizationType.F32
        
        elif n_params <= 1024:
            # very small tensors
            data_qtype = gguf.GGMLQuantizationType.F32
        
        elif n_dims == 4:
            if min(data.shape[:2]) == 4: # output tensor
                data_qtype = fallback
            elif data_shape[-1] == 3: # 3x3 kernel
                data_qtype = fallback
            elif data_shape[-1] == 1: # 1x1 kernel
                #data = np.squeeze(data) # don't do this
                data_qtype = fallback

        # TODO: find keys to keep in higher precision(s) / qtypes
        # if "time_emb_proj.weight" in key:
        #     data_qtype = gguf.GGMLQuantizationType.F16
        # if ".to_v.weight" in key or ".to_out" in key:
        #     data_qtype = gguf.GGMLQuantizationType.F16
        # if "ff.net" in key:
        #     data_qtype = gguf.GGMLQuantizationType.F16

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except gguf.QuantError as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)
        except AttributeError as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        assert len(key) < 64, f"Invalid key length! Cannot store in gguf file. {key}"
        new_name = key # do we need to rename?

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)

warning = """
######################################################
      The quantized file format needs more work.
Consider **not** uploading the resulting files for now
######################################################
"""

if __name__ == "__main__":
    args = parse_args()
    writer, state_dict = load_model(args)
    
    handle_metadata(args, writer, state_dict)
    handle_tensors(args, writer, state_dict)

    writer.write_header_to_file(path=(args.dst or "test.gguf"))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(warning)
