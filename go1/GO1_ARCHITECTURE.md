# GO-1 Model Architecture (Code-Oriented Overview)

GO-1 is a **vision-language-action** policy model composed of four main blocks:

1. **Vision encoder** (`InternVisionModel`)
2. **Language backbone** (`InternLM2ForCausalLMGO1`)
3. **Action Expert** (`ActionExpertModel`)
4. **(Optional) Latent Planner** (`LatentPlannerModel`)

The top-level class is `GO1Model`, which wires these parts together.

## 1) High-level composition

`GO1ModelConfig` defines GO-1 as a composed model with:
- `vision_config`
- `llm_config`
- `action_config`
- `latent_planner_config`
- diffusion scheduler settings (`noise_scheduler_config`)

In code, this composition is explicit in `GO1Model.__init__`, where the vision model, language model, action expert, and optional latent planner are instantiated.

## 2) Vision-to-language bridge

GO-1 uses a ViT to encode images, then converts visual tokens into language-token-space embeddings:

- Extract ViT features (`extract_feature`)
- Remove class token
- Reshape to feature map
- Apply `pixel_shuffle` downsampling (`downsample_ratio`)
- Project with `mlp1` into LLM hidden size

During `common_process`, these projected visual embeddings replace special image context token positions in the language input embeddings (`img_context_token_id`).

## 3) Language backbone as world/context encoder

After image-token injection, GO-1 runs the language backbone (`InternLM2ForCausalLMGO1`) to produce normal language outputs and, crucially, per-layer attention KV cache (`past_key_values`).

Those KV tensors are then projected layer-wise by `k_proj_layers` and `v_proj_layers` from LLM head-dim to Action Expert head-dim (or latent head-dim when latent planning is enabled). This is the main conditioning path from VLM to control.

## 4) Diffusion-based Action Expert

Action prediction is formulated as denoising diffusion over action chunks:

- **State tokens**: robot state is embedded by `state_adaptor`
- **Action tokens**: noisy action trajectory is embedded by `action_adaptor`
- **Condition tokens**: timestep embedding (`time_embedder`) + control-frequency embedding (`freq_embedder`)
- Concatenate all as Action Expert input sequence
- Action Expert attends with VLM-conditioned KV
- `final_layer` decodes action tokens to continuous actions

Training path:
- Add noise to ground-truth actions with `DDPMScheduler`
- Predict denoised action (or epsilon depending on config)
- Optimize MSE via `calc_action_diffusion_loss`

Inference path:
- Start from Gaussian noise
- Iteratively denoise using `DPMSolverMultistepScheduler`
- Output an action chunk (`action_chunk_size × action_dim`)

## 5) Optional Latent Planner (GO-1 vs GO-1 Air)

If `latent_planning=True`, GO-1 enables a latent planner module that first transforms VLM conditioning into latent action tokens before final action denoising.

- **GO-1**: latent planner enabled
- **GO-1 Air**: latent planner disabled (lighter/faster variant)

## 6) Why this architecture is practical

- **Strong perception-language prior** from InternVL/InternLM2 stack
- **Control-friendly head** via diffusion Action Expert
- **Temporal robustness** from iterative denoising
- **Modularity**: can freeze/fine-tune components and optionally remove latent planner

In short, GO-1 is a modular VLA system where a vision-language backbone provides rich context, and a diffusion action decoder converts that context plus robot state into executable action trajectories.
