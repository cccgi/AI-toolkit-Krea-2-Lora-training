"""
Generates ai-toolkit YAML training configs (smoke test + full run) for a
Krea-2 RAW character LoRA, using the settings validated in this
project's research:

  - rank/alpha 32/32 (linear only -- Krea-2 has no conv rank/alpha)
  - face-region regional loss weighting (mask_min_value 0.10), the
    single highest-impact fix found across this project's Flux.2 and
    Krea-2 research
  - 2800 steps as the validated baseline for a ~20-image dataset
  - frozen text encoder with cached text embeddings (required to fit
    a 12.9B model + Qwen3-VL-4B encoder in consumer VRAM)
  - qfloat8 quantization for both the transformer and text encoder

See docs/KREA2_FINDINGS.md for the full research trail behind these
defaults, including what did NOT work (rank 16, no masking).
"""
import os

TEMPLATE = """---
job: extension
config:
  name: "{name}"
  process:
    - type: "sd_trainer"
      training_folder: "{training_folder}"
      device: "cuda:0"
      trigger_word: "{trigger}"
      performance_log_every: 50

      network:
        type: "lora"
        linear: {rank}
        linear_alpha: {rank}
        conv: 0
        conv_alpha: 0

      save:
        dtype: "bf16"
        save_every: {save_every}
        max_step_saves_to_keep: {max_saves}
        save_format: "diffusers"
        push_to_hub: false

      datasets:
        - folder_path: "{dataset_train_dir}"
          mask_path: "{dataset_mask_dir}"
          mask_min_value: 0.10
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          default_caption: "{trigger} {class_noun}"
          cache_latents_to_disk: true
          resolution: [{resolution}]

      train:
        batch_size: 1
        steps: {steps}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        cache_text_embeddings: true
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        timestep_type: "sigmoid"
        optimizer: "adamw8bit"
        optimizer_params:
          weight_decay: 0.01
        lr: 1e-4
        dtype: "bf16"
        skip_first_sample: true
        force_first_sample: false
        disable_sampling: {disable_sampling}
        ema_config:
          use_ema: false
        loss_type: "mse"

      model:
        name_or_path: "krea/Krea-2-Raw"
        arch: "krea2"
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
        low_vram: true
        model_kwargs:
          text_encoder_path: "Qwen/Qwen3-VL-4B-Instruct"
          vae_path: "Qwen/Qwen-Image"

      sample:
        sampler: "flowmatch"
        sample_every: {sample_every}
        width: {resolution}
        height: {resolution}
        seed: 104729
        walk_seed: false
        guidance_scale: 3.5
        sample_steps: {sample_steps}
        neg: "airbrushed, plastic skin, waxy, over-retouched, beauty filter"
        prompts:
{prompts}

      logging:
        log_every: 10
meta:
  name: "[name]"
  version: "2.0"
"""

DEFAULT_PROMPTS = [
    "[trigger] {class_noun}, editorial head-and-shoulders portrait, neutral studio backdrop, soft window light, looking at camera",
    "[trigger] {class_noun}, seated at a cafe window holding a coffee cup, candid documentary photograph, overcast daylight",
    "[trigger] {class_noun}, wearing a dark blazer, waist-up professional portrait, deep red backdrop, soft even studio lighting",
    "[trigger] {class_noun}, autumn park portrait, holding yellow leaves, natural daylight, medium shot",
]


def _format_prompts(class_noun):
    lines = []
    for p in DEFAULT_PROMPTS:
        lines.append(f'          - "{p.format(class_noun=class_noun)}"')
    return "\n".join(lines)


def generate_configs(
    subject_name,
    trigger,
    class_noun,
    dataset_train_dir,
    dataset_mask_dir,
    output_dir,
    configs_dir,
    rank=32,
    steps=2800,
    resolution=1024,
):
    """Writes smoke and full YAML configs to configs_dir; returns their paths."""
    os.makedirs(configs_dir, exist_ok=True)
    prompts_block = _format_prompts(class_noun)

    full_name = f"{subject_name}_krea2_raw_r{rank}_{steps}"
    smoke_name = f"{full_name}_smoke"

    common = dict(
        trigger=trigger,
        class_noun=class_noun,
        dataset_train_dir=dataset_train_dir.replace("\\", "\\\\"),
        dataset_mask_dir=dataset_mask_dir.replace("\\", "\\\\"),
        training_folder=output_dir.replace("\\", "\\\\"),
        rank=rank,
        resolution=resolution,
        prompts=prompts_block,
    )

    full_yaml = TEMPLATE.format(
        name=full_name,
        steps=steps,
        save_every=400,
        max_saves=8,
        sample_every=400,
        sample_steps=32,
        disable_sampling="false",
        **common,
    )
    smoke_yaml = TEMPLATE.format(
        name=smoke_name,
        steps=1,
        save_every=1,
        max_saves=1,
        sample_every=1,
        sample_steps=8,
        disable_sampling="true",
        **common,
    )

    full_path = os.path.join(configs_dir, f"{full_name}.yaml")
    smoke_path = os.path.join(configs_dir, f"{smoke_name}.yaml")

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(full_yaml)
    with open(smoke_path, "w", encoding="utf-8") as f:
        f.write(smoke_yaml)

    return smoke_path, full_path
