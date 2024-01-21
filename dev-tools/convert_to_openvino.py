from typing import Dict, Optional, Tuple, OrderedDict
from transformers import CLIPTextConfig
from diffusers import UNet2DConditionModel

import torch

from optimum.exporters.onnx.model_configs import VisionOnnxConfig, NormalizedConfig, DummyVisionInputGenerator, DummyTimestepInputGenerator, DummySeq2SeqDecoderTextInputGenerator, DummySeq2SeqDecoderTextInputGenerator
from optimum.exporters.openvino import main_export
from optimum.utils.input_generators import DummyInputGenerator, DEFAULT_DUMMY_SHAPES
from optimum.utils.normalized_config import NormalizedTextConfig

# IMPORTANT: You need to specify some scheduler in downloaded model cache folder to avoid errors

class CustomDummyTimestepInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "timestep",
        "timestep_cond",
        "text_embeds",
        "time_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        time_cond_proj_dim: int = 256,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        self.text_encoder_projection_dim = normalized_config.text_encoder_projection_dim
        self.time_ids = 5 if normalized_config.requires_aesthetics_score else 6
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        self.time_cond_proj_dim = normalized_config.get("time_cond_proj_dim", time_cond_proj_dim)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size]

        if input_name == "timestep":
            return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=int_dtype)

        if input_name == "timestep_cond":
            shape.append(self.time_cond_proj_dim)
            return self.random_float_tensor(shape, min_value=-1.0, max_value=1.0, framework=framework, dtype=float_dtype)


        shape.append(self.text_encoder_projection_dim if input_name == "text_embeds" else self.time_ids)
        return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)

class LCMUNetOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    # The ONNX export of a CLIPText architecture, an other Stable Diffusion component, needs the Trilu
    # operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        CustomDummyTimestepInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = OrderedDict({
            "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "timestep": {0: "steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "timestep_cond": {0: "batch_size"},
        })

        # TODO : add text_image, image and image_embeds
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs["text_embeds"] = {0: "batch_size"}
            common_inputs["time_ids"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return {
            "sample": "out_sample",
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            dummy_inputs["added_cond_kwargs"] = {
                "text_embeds": dummy_inputs.pop("text_embeds"),
                "time_ids": dummy_inputs.pop("time_ids"),
            }

        return dummy_inputs

    def ordered_inputs(self, model) -> Dict[str, Dict[int, str]]:
        return self.inputs # Breaks order if timestep_cond involved ( so just copy original one )

model_id = "SimianLuo/LCM_Dreamshaper_v7"

text_encoder_config = CLIPTextConfig.from_pretrained(model_id, subfolder = "text_encoder")
unet_config = UNet2DConditionModel.from_pretrained(model_id, subfolder = "unet").config

unet_config.text_encoder_projection_dim = text_encoder_config.projection_dim
unet_config.requires_aesthetics_score = False

custom_onnx_configs = {
    "unet": LCMUNetOnnxConfig(config = unet_config, task = "semantic-segmentation")
}

main_export(model_name_or_path = model_id, output = "./", task = "stable-diffusion", fp16 = False, int8 = False, custom_onnx_configs = custom_onnx_configs)
