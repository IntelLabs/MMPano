import os
from typing import Union, Optional

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionUpscalePipeline


def is_on_hpu(device: str) -> bool:
    """ Return True is the device is a Gaudi/HPU device.
    """
    return "hpu" in device


def get_datatype(data_type: Union[str, torch.dtype]):
    if isinstance(data_type, torch.dtype):
        return data_type
    if data_type in ["fp32", "float32"]:
        return torch.float
    elif data_type in ["fp16", "float16"]:
        return torch.float16
    elif data_type in ["bfloat16", "bf16"]:
        return torch.bfloat16
    else:
        raise RuntimeError(f"Got unknown dtype {data_type}")


def optimize_stable_diffusion_pipeline(pipeline, device, datatype, cpu_offload: bool = False, enable_xformers: bool = False):
    pipeline.to(device)
    if is_on_hpu(device):
        assert datatype in ["bfloat16", "float32"] or datatype in [torch.bfloat16, torch.float32]
        pass
    else:
        # Cuda
        # TODO(Joey): Check if there is an Intel version of xformers
        # pipeline.unet = torch.compile(pipeline.unet)
        if enable_xformers:
            pipeline.set_use_memory_efficient_attention_xformers(enable_xformers)

    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_model_cpu_offload()
    return pipeline.to(device)


def optimize_blip(model, device, datatype):
    model.to(device)
    return model


def load_diffusion_model(model_name: str = "stabilityai/stable-diffusion-2-inpainting",
                         device: str = "cuda",
                         dtype: Union[str, torch.dtype] = "float16",
                         cpu_offload: bool = False):
    """ Load diffusion or diffusion inpainting model for text-to-image.
    """
    print(f"Loading text-to-image model {model_name} ...")

    torch_dtype = get_datatype(dtype)

    if is_on_hpu(device=device):
        assert dtype in ["bfloat16", "float32"] or dtype in [torch.bfloat16, torch.float32]
        from optimum.habana.diffusers import GaudiDDIMScheduler
        if "inpaint" in model_name:
            from optimum.habana.diffusers import GaudiStableDiffusionInpaintPipeline as DiffusionPipelineClass
        else:
            from optimum.habana.diffusers import GaudiStableDiffusionPipeline as DiffusionPipelineClass

        # Load model and scheduler on Gaudi/HPU
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        kwargs = {
            "scheduler": scheduler,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion"
        }
        pipe = DiffusionPipelineClass.from_pretrained(model_name, **kwargs).to(torch_dtype)
    else:
        # Load model and scheduler
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # TODO(Joey): Check why enable xformers + torch.compile for inpainting model gets error
    pipe = optimize_stable_diffusion_pipeline(pipe, device, torch_dtype, cpu_offload, enable_xformers=False)
    return pipe


def load_blip_model_and_processor(model_name: str = "Salesforce/blip2-flan-t5-xl",  # "Salesforce/blip2-opt-2.7b"
                                  device: str = "cuda",
                                  dtype: Union[str, torch.dtype] = "float16"):
    """ Load BLIP model for image-to-text.
    """
    print(f"Loading image-to-text model {model_name} ...")

    torch_dtype = get_datatype(dtype)

    if "blip2" in model_name:
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration
    else:
        # Blip
        assert "blip" in model_name
        processor_class = BlipProcessor
        model_class = BlipForConditionalGeneration

    if is_on_hpu(device=device):
        assert dtype in ["bfloat16", "float32"] or dtype in [torch.bfloat16, torch.float32]
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        # TODO(Joey): Check optimum-habana once it has Blip2 support
        adapt_transformers_to_gaudi()
    else:
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration

    # Get a blip description
    processor = processor_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = optimize_blip(model, device, torch_dtype)

    return processor, model


def load_upscaler_model(model_name: str = "stabilityai/stable-diffusion-x4-upscaler",
                        device: str = "cuda",
                        dtype: Union[str, torch.dtype] = "float16",
                        cpu_offload: bool = False):
    """ Load super resolution model for upscaling.
    """
    print(f"Loading super resolution model {model_name} ...")

    torch_dtype = get_datatype(dtype)

    if is_on_hpu(device=device):
        assert dtype in ["bfloat16", "float32"] or dtype in [torch.bfloat16, torch.float32]
        from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionUpscalePipeline

        # Load model and scheduler on Gaudi/HPU
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        kwargs = {
            "scheduler": scheduler,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion"
        }
        pipe = GaudiStableDiffusionUpscalePipeline.from_pretrained(model_name, **kwargs).to(torch_dtype)
    else:
        # Load model and scheduler
        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        # TODO(Joey): Check why enable xformers + torch.compiler for inpainting model gets error
    pipe = optimize_stable_diffusion_pipeline(pipe, device, torch_dtype, cpu_offload, enable_xformers=False)

    return pipe


def load_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
             device: str = "cuda",
             dtype: Optional[Union[str, torch.dtype]] = "float16",
             trust_remote_code: bool = False,
             hf_token: Optional[str] = None):
    """ Load LLM model.
    """
    print(f"Loading LLM {model_name} ...")

    if is_on_hpu(device=device):
        assert dtype in ["bfloat16", "float32"] or dtype in [torch.bfloat16, torch.float32]
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=hf_token).eval().to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def release_memory(model, tokenizer, device: str = "cuda"):
    import gc
    del tokenizer
    del model

    if device == "cuda":
        torch.cuda.empty_cache()
    else:
        # TODO(Tien Pei Chou): Add Gaudi and XPU
        raise NotImplementedError()
    # accelerator.free_memory()
    gc.collect()
