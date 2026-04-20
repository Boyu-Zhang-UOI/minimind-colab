<map version="1.0.1">
<!-- MiniMind Model Architecture Mind Map -->
<!-- Open with: FreeMind, Freeplane, XMind, MindMeister, or any FreeMind-compatible tool -->
<node TEXT="MiniMind" FOLDED="false" COLOR="#000000">
<font NAME="SansSerif" SIZE="20" BOLD="true"/>
<richcontent TYPE="NOTE"><html><head/><body><p>MiniMind: A lightweight causal language model built with PyTorch and HuggingFace Transformers.<br/>Default: ~64M params, hidden_size=768, 8 layers, 8 heads (4 KV), vocab=6400.</p></body></html></richcontent>

<!-- ==================== Configuration ==================== -->
<node TEXT="Configuration" POSITION="right" FOLDED="false" COLOR="#0033CC">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="configure"/>
<richcontent TYPE="NOTE"><html><head/><body><p>MiniMindConfig (model/model_minimind.py)<br/>Extends PretrainedConfig from HuggingFace Transformers.</p></body></html></richcontent>

<node TEXT="Model Dimensions" FOLDED="false" COLOR="#0066CC">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="hidden_size: 768" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="num_hidden_layers: 8" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="vocab_size: 6400" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="intermediate_size: ceil(768*pi/64)*64" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>

<node TEXT="Attention Config" FOLDED="false" COLOR="#0066CC">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="num_attention_heads: 8" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="num_key_value_heads: 4 (GQA)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="head_dim: hidden_size / num_heads = 96" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="flash_attn: True" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>

<node TEXT="Position Encoding" FOLDED="false" COLOR="#0066CC">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="max_position_embeddings: 32768" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="rope_theta: 1e6" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="YaRN Scaling (optional)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<node TEXT="factor: 16" COLOR="#666666"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="original_max_position_embeddings: 2048" COLOR="#666666"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="type: yarn" COLOR="#666666"><font NAME="SansSerif" SIZE="11"/></node>
</node>
</node>

<node TEXT="MoE Config (optional)" FOLDED="false" COLOR="#0066CC">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="use_moe: False (default)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="num_experts: 4" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="num_experts_per_tok: 1" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="norm_topk_prob: True" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="router_aux_loss_coef: 5e-4" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>

<node TEXT="Training Config" FOLDED="false" COLOR="#0066CC">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="dropout: 0.0" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="hidden_act: silu" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="rms_norm_eps: 1e-6" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>
</node>

<!-- ==================== Model Architecture ==================== -->
<node TEXT="Model Architecture" POSITION="right" FOLDED="false" COLOR="#CC0000">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="idea"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Core model classes in model/model_minimind.py<br/>Follows a decoder-only Transformer architecture.</p></body></html></richcontent>

<node TEXT="MiniMindForCausalLM" FOLDED="false" COLOR="#CC3333">
<font NAME="SansSerif" SIZE="14" BOLD="true"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Top-level class. Extends PreTrainedModel + GenerationMixin.<br/>Handles loss computation (cross-entropy) and text generation.<br/>Weight tying: lm_head.weight = embed_tokens.weight.</p></body></html></richcontent>

<node TEXT="MiniMindModel" FOLDED="false" COLOR="#CC3333">
<font NAME="SansSerif" SIZE="13" BOLD="true"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Core transformer model. Contains embedding, N transformer blocks, and final norm.<br/>Precomputes RoPE frequency buffers at init.</p></body></html></richcontent>

<node TEXT="embed_tokens (nn.Embedding)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Token embedding layer: vocab_size x hidden_size.<br/>Weights are tied with lm_head.</p></body></html></richcontent>
</node>

<node TEXT="dropout (nn.Dropout)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
</node>

<node TEXT="layers x8 (MiniMindBlock)" FOLDED="false" COLOR="#CC6666">
<font NAME="SansSerif" SIZE="13" BOLD="true"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Stack of N=8 transformer blocks.<br/>Each block: Pre-Norm Attention + Residual, then Pre-Norm FFN + Residual.</p></body></html></richcontent>

<node TEXT="input_layernorm (RMSNorm)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Pre-attention normalization.<br/>RMSNorm: weight * x / sqrt(mean(x^2) + eps).</p></body></html></richcontent>
</node>

<node TEXT="self_attn (Attention)" FOLDED="false" COLOR="#CC6666">
<font NAME="SansSerif" SIZE="13"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Grouped Query Attention (GQA): 8 query heads, 4 KV heads.<br/>Supports Flash Attention and manual attention with causal mask.<br/>Includes KV cache for efficient autoregressive generation.</p></body></html></richcontent>

<node TEXT="q_proj (Linear: hidden -&gt; n_heads * head_dim)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="k_proj (Linear: hidden -&gt; n_kv_heads * head_dim)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="v_proj (Linear: hidden -&gt; n_kv_heads * head_dim)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="q_norm / k_norm (RMSNorm per head)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="apply_rotary_pos_emb (RoPE)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Rotary Position Embedding applied to Q and K.<br/>Uses precomputed cos/sin frequency buffers.<br/>rotate_half pattern for efficient computation.</p></body></html></richcontent>
</node>
<node TEXT="repeat_kv (KV head expansion)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Repeats KV heads to match query head count for GQA.<br/>n_rep = num_attention_heads / num_key_value_heads = 2.</p></body></html></richcontent>
</node>
<node TEXT="Flash Attention / Manual Attention" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Flash: Uses F.scaled_dot_product_attention (causal).<br/>Manual: QK^T/sqrt(d) + causal mask + softmax + V.</p></body></html></richcontent>
</node>
<node TEXT="o_proj (Linear: n_heads * head_dim -&gt; hidden)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>

<node TEXT="+ Residual Connection" COLOR="#999999">
<font NAME="SansSerif" SIZE="12" ITALIC="true"/>
</node>

<node TEXT="post_attention_layernorm (RMSNorm)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
</node>

<node TEXT="mlp (FeedForward or MOEFeedForward)" FOLDED="false" COLOR="#CC6666">
<font NAME="SansSerif" SIZE="13"/>

<node TEXT="FeedForward (SwiGLU)" FOLDED="false" COLOR="#CC9999">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x)).<br/>Used when use_moe=False (default).</p></body></html></richcontent>
<node TEXT="gate_proj (Linear: hidden -&gt; intermediate)" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="up_proj (Linear: hidden -&gt; intermediate)" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="SiLU activation" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="down_proj (Linear: intermediate -&gt; hidden)" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
</node>

<node TEXT="MOEFeedForward (Mixture of Experts)" FOLDED="false" COLOR="#CC9999">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Used when use_moe=True.<br/>Router gate selects top-K experts per token.<br/>Includes auxiliary load-balancing loss.</p></body></html></richcontent>
<node TEXT="gate (Linear: hidden -&gt; num_experts)" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="experts x4 (FeedForward instances)" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="Top-K expert selection (k=1)" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="Weighted expert output aggregation" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
<node TEXT="Auxiliary load-balancing loss" COLOR="#333333"><font NAME="SansSerif" SIZE="11"/></node>
</node>
</node>

<node TEXT="+ Residual Connection" COLOR="#999999">
<font NAME="SansSerif" SIZE="12" ITALIC="true"/>
</node>
</node>

<node TEXT="norm (final RMSNorm)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
</node>

<node TEXT="freqs_cos / freqs_sin (RoPE buffers)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Precomputed at init via precompute_freqs_cis().<br/>Supports YaRN scaling for extended context length.</p></body></html></richcontent>
</node>
</node>

<node TEXT="lm_head (Linear: hidden -&gt; vocab)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Language model head for next-token prediction.<br/>Weight tied with embed_tokens.</p></body></html></richcontent>
</node>

<node TEXT="generate() method" FOLDED="false" COLOR="#CC3333">
<font NAME="SansSerif" SIZE="13"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Autoregressive text generation with KV caching.<br/>Custom implementation with full control over decoding.</p></body></html></richcontent>
<node TEXT="Temperature scaling" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="Top-K filtering" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="Top-P (nucleus) sampling" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="Repetition penalty" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="KV cache reuse" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="Streaming support (TextStreamer)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>
</node>
</node>

<!-- ==================== LoRA ==================== -->
<node TEXT="LoRA Adaptation" POSITION="right" FOLDED="false" COLOR="#009900">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="attach"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Low-Rank Adaptation in model/model_lora.py.<br/>Applied to square Linear layers (in_features == out_features).</p></body></html></richcontent>

<node TEXT="LoRA Module" FOLDED="false" COLOR="#339933">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="Matrix A (Linear: in -&gt; rank, Gaussian init)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="Matrix B (Linear: rank -&gt; out, zero init)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="Output: B(A(x)), added to original W(x)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>

<node TEXT="API Functions" FOLDED="false" COLOR="#339933">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="apply_lora(model, rank=16)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Attaches LoRA modules to all square Linear layers.<br/>Monkey-patches forward to: original(x) + lora(x).</p></body></html></richcontent>
</node>
<node TEXT="load_lora(model, path)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="save_lora(model, path)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="merge_lora(model, lora_path, save_path)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Merges LoRA weights: W' = W + B@A.<br/>Produces a new checkpoint file (not in-place).</p></body></html></richcontent>
</node>
</node>
</node>

<!-- ==================== Training Pipeline ==================== -->
<node TEXT="Training Pipeline" POSITION="left" FOLDED="false" COLOR="#FF6600">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="launch"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Training scripts in trainer/ directory.<br/>All use cosine LR schedule, DDP support, and checkpoint resume.</p></body></html></richcontent>

<node TEXT="Tokenizer Training" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>train_tokenizer.py: Trains BPE tokenizer (vocab=6400).</p></body></html></richcontent>
</node>

<node TEXT="Pretraining" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>train_pretrain.py: Next-token prediction on raw text.<br/>Uses PretrainDataset with BOS/EOS wrapping.</p></body></html></richcontent>
</node>

<node TEXT="Supervised Fine-Tuning" FOLDED="false" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="Full SFT (train_full_sft.py)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Full-parameter fine-tuning on chat data.<br/>Uses SFTDataset with selective label masking.</p></body></html></richcontent>
</node>
<node TEXT="LoRA SFT (train_lora.py)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Parameter-efficient fine-tuning.<br/>Freezes base model, only trains LoRA weights.</p></body></html></richcontent>
</node>
</node>

<node TEXT="Alignment / RLHF" FOLDED="false" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="DPO (train_dpo.py)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Direct Preference Optimization.<br/>Uses DPODataset with chosen/rejected pairs.</p></body></html></richcontent>
</node>
<node TEXT="PPO (train_ppo.py)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Proximal Policy Optimization with reward model.</p></body></html></richcontent>
</node>
<node TEXT="GRPO (train_grpo.py)" COLOR="#333333">
<font NAME="SansSerif" SIZE="12"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Group Relative Policy Optimization.<br/>Uses RLAIFDataset with thinking support.</p></body></html></richcontent>
</node>
</node>

<node TEXT="Knowledge Distillation" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>train_distillation.py: Distills knowledge from a larger teacher model.</p></body></html></richcontent>
</node>

<node TEXT="Agent / Tool Use" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>train_agent.py: RL training for tool-calling capabilities.<br/>Uses AgentRLDataset and rollout_engine.py.</p></body></html></richcontent>
</node>

<node TEXT="Utilities (trainer_utils.py)" FOLDED="false" COLOR="#FF9933">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="get_lr(): Cosine LR schedule" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="init_distributed_mode(): DDP setup" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="lm_checkpoint(): Save/load with resume" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="init_model(): Model + tokenizer init" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="SkipBatchSampler: Resume-aware sampling" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>
</node>

<!-- ==================== Dataset ==================== -->
<node TEXT="Datasets" POSITION="left" FOLDED="false" COLOR="#9900CC">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="list"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Dataset classes in dataset/lm_dataset.py.<br/>All extend torch.utils.data.Dataset.</p></body></html></richcontent>

<node TEXT="PretrainDataset" COLOR="#AA33CC">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Raw text for next-token prediction.<br/>Wraps with BOS/EOS, pads to max_length=512.</p></body></html></richcontent>
</node>

<node TEXT="SFTDataset" COLOR="#AA33CC">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Chat conversations with selective label masking.<br/>Only assistant responses are supervised (labels != -100).<br/>Supports tool calls, system prompts, thinking tags.</p></body></html></richcontent>
</node>

<node TEXT="DPODataset" COLOR="#AA33CC">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Chosen/rejected response pairs for preference optimization.<br/>Generates loss masks for assistant tokens only.</p></body></html></richcontent>
</node>

<node TEXT="RLAIFDataset" COLOR="#AA33CC">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Prompts for RL-based training (GRPO/PPO).<br/>Probabilistic thinking mode (50% ratio).</p></body></html></richcontent>
</node>

<node TEXT="AgentRLDataset" COLOR="#AA33CC">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Tool-calling conversations with ground-truth answers.<br/>Used for agent RL training.</p></body></html></richcontent>
</node>

<node TEXT="Data Processing" FOLDED="false" COLOR="#AA33CC">
<font NAME="SansSerif" SIZE="14"/>
<node TEXT="pre_processing_chat(): Add random system prompts (20%)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
<node TEXT="post_processing_chat(): Remove empty think tags (80%)" COLOR="#333333"><font NAME="SansSerif" SIZE="12"/></node>
</node>
</node>

<!-- ==================== Inference & Evaluation ==================== -->
<node TEXT="Inference &amp; Evaluation" POSITION="left" FOLDED="false" COLOR="#006666">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="xmag"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Scripts in scripts/ and root directory for inference and serving.</p></body></html></richcontent>

<node TEXT="eval_llm.py" COLOR="#339999">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Interactive chat interface for model evaluation.<br/>Supports auto-test mode with preset prompts and manual input mode.<br/>Shows decode speed (tokens/s).</p></body></html></richcontent>
</node>

<node TEXT="serve_openai_api.py" COLOR="#339999">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>OpenAI-compatible API server for integration.</p></body></html></richcontent>
</node>

<node TEXT="web_demo.py" COLOR="#339999">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Web-based demo for interactive model testing.</p></body></html></richcontent>
</node>

<node TEXT="eval_toolcall.py" COLOR="#339999">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Evaluation of tool-calling capabilities.</p></body></html></richcontent>
</node>

<node TEXT="convert_model.py" COLOR="#339999">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Convert between PyTorch and HuggingFace model formats.</p></body></html></richcontent>
</node>
</node>

<!-- ==================== Key Techniques ==================== -->
<node TEXT="Key Techniques" POSITION="left" FOLDED="false" COLOR="#996600">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="bookmark"/>

<node TEXT="RoPE (Rotary Position Embedding)" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>precompute_freqs_cis() + apply_rotary_pos_emb().<br/>Encodes relative position via rotation in complex plane.<br/>Supports YaRN extension for longer contexts.</p></body></html></richcontent>
</node>

<node TEXT="GQA (Grouped Query Attention)" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>8 query heads share 4 KV heads (n_rep=2).<br/>Reduces KV cache memory by 2x vs standard MHA.<br/>repeat_kv() expands KV for attention computation.</p></body></html></richcontent>
</node>

<node TEXT="SwiGLU Activation" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>FeedForward: down(silu(gate(x)) * up(x)).<br/>Gated variant of SiLU for better training dynamics.</p></body></html></richcontent>
</node>

<node TEXT="RMSNorm" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Root Mean Square Layer Normalization.<br/>Simpler and faster than LayerNorm (no mean subtraction).<br/>Applied pre-attention and pre-FFN (Pre-Norm architecture).</p></body></html></richcontent>
</node>

<node TEXT="Flash Attention" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Uses PyTorch F.scaled_dot_product_attention.<br/>Falls back to manual attention when needed (KV cache, masking).</p></body></html></richcontent>
</node>

<node TEXT="Weight Tying" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>embed_tokens.weight = lm_head.weight.<br/>Reduces parameters and improves training stability.</p></body></html></richcontent>
</node>

<node TEXT="Mixture of Experts (MoE)" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Optional MoE with Top-K routing and auxiliary loss.<br/>4 experts, 1 active per token (default).<br/>Increases model capacity without proportional compute cost.</p></body></html></richcontent>
</node>

<node TEXT="KV Cache" COLOR="#CC9933">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Caches key/value tensors for efficient autoregressive generation.<br/>Only new tokens processed at each step.</p></body></html></richcontent>
</node>
</node>

<!-- ==================== Notebooks ==================== -->
<node TEXT="Course Notebooks" POSITION="right" FOLDED="false" COLOR="#CC0066">
<font NAME="SansSerif" SIZE="16" BOLD="true"/>
<icon BUILTIN="edit"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Educational Jupyter notebooks in notebooks/ directory.<br/>6 topics with regular + LiveCoding variants (12 total).<br/>Additional step-by-step and session variants.</p></body></html></richcontent>

<node TEXT="Notebook 1: Tokenizer" COLOR="#CC3399">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>BPE tokenizer training, encoding/decoding, chat templates.</p></body></html></richcontent>
</node>

<node TEXT="Notebook 2: Model Architecture" COLOR="#CC3399">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>RMSNorm, RoPE, Attention (GQA), FeedForward (SwiGLU),<br/>MoE, Transformer Block, full model assembly.</p></body></html></richcontent>
</node>

<node TEXT="Notebook 3: Pretraining" COLOR="#CC3399">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Next-token prediction, data loading, training loop.</p></body></html></richcontent>
</node>

<node TEXT="Notebook 4: DPO Alignment" COLOR="#CC3399">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Direct Preference Optimization for alignment.</p></body></html></richcontent>
</node>

<node TEXT="Notebook 5: SFT and LoRA" COLOR="#CC3399">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Supervised fine-tuning, LoRA adaptation, parameter efficiency.</p></body></html></richcontent>
</node>

<node TEXT="Notebook 6: Evaluation and Inference" COLOR="#CC3399">
<font NAME="SansSerif" SIZE="14"/>
<richcontent TYPE="NOTE"><html><head/><body><p>Model evaluation, text generation, decode speed benchmarking.</p></body></html></richcontent>
</node>
</node>

</node>
</map>
