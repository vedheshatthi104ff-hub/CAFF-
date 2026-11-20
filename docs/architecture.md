# Architecture and implementation record

## Paper-based baseline

The baseline is implemented in `src/caff_efficientvpr/models/` as DINOv2-S/14 + SceneVPT + MsIA + OLE + SGFS. It is a reimplementation from the paper and official DINOv2 interface; it is not authors' released code.

| Component | Operation | Shape in this implementation | Status |
|---|---|---|---|
| DINOv2-S/14 | Official pretrained Meta weights; standard no-register variant | Patch tokens `(B,N,384)` and CLS `(B,384)` | Official checkpoint loaded successfully in a CPU forward pass. Hub source code is fetched by PyTorch Hub on first use. |
| SceneVPT | Insert 32 learned prompt tokens after CLS at each block. From previous-block CLS attention, average heads, normalize scores over prompt positions, compute `α=sigmoid(s−γ)`, and mix prior prompts with the next learned prompt table. | Input image 266² gives 361 patch tokens; sequence is `(B,394,384)` at each block. | Equations and learnable γ are paper-specified; threshold initialization `1/32` and exact attention extraction are inferred. |
| MsIA | Three 3×3 convolutions at strides 1, 2, 4; flatten/concatenate; multi-head attention Q=backbone patches, K/V=multi-scale tokens; residual + LayerNorm. | For a 19×19 patch map with same padding: scale grids 19², 10², 5², thus 486 K/V tokens; output `(B,361,384)`. | Scales and Q/K/V roles are paper-specified. Padding, residual, normalization, and convolution widths are implementation choices. |
| OLE | 1×3 horizontal and 3×1 vertical convolutions; orthogonal mode uses GeM + MLP guide, subtracts the normalized projection, then concatenates and 1×1 fuses. `concat` mode bypasses orthogonalization for the supplement ablation. | `361` local tokens at 266 input, each 384-D. | The supplement compares nonlinear orthogonal fusion with direct concatenation. GeM p=3, epsilon, MLP dimensions, padding and fusion widths remain inferred/configurable. |
| SGFS | Prompt self-attention; cross-attention from prompt features to concatenated MsIA/OLE K/V; LayerNorm; linear compression along prompt axis from 32 to 8. | `32×384` prompt features → `8×384`; concatenate CLS gives `(B,3456)`. | Sequence of operations and eight-token compression are paper/supplement supported. Attention residual conventions are inferred from the diagram/equations. |
| Descriptor | Concatenate final normalized CLS with flattened eight SGFS tokens; L2 normalize. | 9×384 = 3456 dimensions. | 3456 is paper-reported and shape-consistent; final L2 normalization is a conventional retrieval choice, not independently verified against unavailable official code. |

With 322×322 evaluation input, DINOv2/14 yields a 23×23 patch map. Same-padding stride-2/4 MsIA outputs become 12×12 and 6×6 in this implementation. The paper reports 322² evaluation resize; it does not specify the convolution padding.

### SceneVPT details

The official DINOv2 block's existing QKV projection is used to read CLS-to-prompt attention without adding a separate attention module. The implementation computes the CLS query's softmax over the complete token sequence and averages the prompt positions over heads. After each block, its prompt states are combined with the next layer's learned prompt table using the paper's adaptive equation. A learnable scalar `gamma` is shared across layers; the paper describes a learnable threshold but does not state whether it is shared or per layer. Sharing it is an inferred, low-parameter choice. The loader rejects random initialization and loads the official no-register `dinov2_vits14` weights from the bundled checkpoint or official Hub fallback.

### CAFF extension (v2: gate inside the baseline path)

CAFF is optional and instantiated only when `use_caff=true`. It receives no second backbone pass.

- Streams: global = MsIA tokens, local = OLE tokens (the same two streams the baseline concatenates for SGFS).
- Condition vector: DINOv2 CLS token, mean-pooled MsIA tokens and mean-pooled OLE tokens, concatenated and LayerNorm-ed.
- Gate: small MLP -> two-way softmax (`w_global`, `w_local`, nonnegative, sum to one). The last layer is zero-initialised, so `w = 0.5/0.5`.
- Application: each stream is multiplied by `2 * w` before SGFS (1.0 at initialisation). SGFS input, CLS token, SGFS compression and the (8 + 1) x 384 = 3456-D descriptor layout are identical to the baseline, so CAFF equals the baseline at step 0 (covered by `test_caff_matches_baseline_at_initialization`).
- Controls: `caff_gate_mode="static"` learns two image-independent weights (isolates the benefit of conditioning); `"equal"` is a no-op equal to the baseline.

Why v2: the first CAFF head mean-pooled the tokens, projected each stream to 128-D and expanded 256-D to 3456-D with a random linear layer. That capped descriptor rank at 256, dropped the CLS token and changed the SGFS inputs (OLE-only K/V), so the short pilot (R@1 39.1% vs 51.8% baseline) confounded those changes with the gating itself. v1 pilot numbers should not be compared with v2. This is a project-proposed hypothesis, not an EfficientVPR equation.

### Parameter counts and tensor verification

The code exposes `model.parameter_counts()` for actual total/trainable counts after loading the selected configuration. The CPU baseline load reported 29,437,705 total/trainable parameters. SceneVPT's prompt table alone has 12×32×384 = 147,456 parameters for standard DINOv2-S, plus the threshold; this is a shape-derived count, not a whole-model count. The supplement's 0.15M prompt-parameter report is consistent with that scale.

