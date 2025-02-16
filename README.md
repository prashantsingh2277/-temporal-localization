# -temporal-localization
Temporal Action Localization with a Custom Deep Learning Architecture

# Comparison of State-of-the-Art Models and Custom Model Design

## State-of-the-Art Models

### TCANet
TCANet introduces a **Local‑Global Temporal Encoder (LGTE)** that divides the input feature channels into groups—some for local modeling (using operations similar to dynamic convolutions) and some for global context (using long‑range interactions). It then fuses these representations and uses a **Temporal Boundary Regressor (TBR)** that combines both frame-level (boundary) and segment-level (center/duration) regressions in an end-to-end manner. This design is very effective in precisely refining action boundaries while keeping computation efficient.

### Proposal Relation Network (PRN)
PRN builds on BMN by embedding a **proposal relation module** that uses self-attention to explicitly capture dependencies among proposals. By modeling the relations between candidate action segments, PRN improves the quality and accuracy of proposal generation. Its design emphasizes the importance of contextual relations among proposals, which are often overlooked in simpler boundary regression methods.

### Video Mamba Suite
Video Mamba Suite leverages **state space models (SSMs)** as a versatile and efficient alternative to transformers for video understanding. It explores various roles for SSMs (including as temporal models and adapters) and achieves linear complexity with strong performance. While the SSM-based models (like Mamba or its variants) can be very efficient for long sequence modeling, they are often more complex to implement and tune compared to convolutional or attention-based methods.

## Selection and Design Criteria for the Custom Model

### Efficiency
- **LocalTemporalBlock:** Uses a lightweight 1D convolution, which is computationally inexpensive and effective for capturing fine-grained local motion.
- **GlobalContextBlock:** Uses a multihead attention mechanism on a reduced channel subset (half the channels) to capture longer-range dependencies without a heavy computational burden.

### Multi-granularity Fusion
- Inspired by TCANet’s approach, the model fuses local and global features. This enables the network to maintain sensitivity to subtle boundary details (via local processing) while also understanding the overall action context (via global processing).

### Unified Regression Framework
- By using separate regression heads for boundary and segment predictions, the model mirrors the complementary strategy seen in TCANet and PRN. This unified regression approach is key to addressing the inherent challenges of temporal localization—precisely identifying both the boundaries and the overall extent of an action.

### Scalability
- Our custom model is simple enough to be trained on a smaller dataset and within a reasonable time frame.
- In contrast, while models like Video Mamba Suite show strong performance and efficiency in handling long sequences, they tend to be more complex and require more careful tuning.
- We opted for a more straightforward design that still leverages the advantages of both local and global processing.

---

This custom architecture balances simplicity and effectiveness by drawing inspiration from TCANet, PRN, and Video Mamba Suite. It captures local details and global context through dedicated branches, fuses them, and then uses unified regression heads to predict action boundaries and segments. The resulting model is efficient, scalable, and well-suited for training on a smaller dataset while still achieving meaningful temporal localization.
