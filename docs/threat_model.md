# Threat Model

## Defender Model

The defender controls the SSL pretraining pipeline:
- Choice of augmentations, sampling strategy, loss function, optimizer
- Access to training data (but **no labels** during pretraining)
- Ability to modify training dynamics (e.g., sample weighting, gradient clipping)
- Does **not** know which specific samples are poisoned
- Does **not** require a separate clean validation set (fully unsupervised defense)

## Attacker Model

The attacker can inject a small fraction ρ of poisoned samples:
- **Capability**: modify a fraction ρ ∈ {0.5%, 1%, 3%, 5%, 10%} of the training corpus
- **Objective**: the pretrained encoder produces representations that cause downstream classifiers to misclassify triggered inputs as a target class
- **Constraint**: clean utility must remain high (otherwise the poisoned model would be rejected)

### Attack families evaluated

| Attack | Mechanism | Trigger Type |
|--------|-----------|--------------|
| Patch Backdoor (BadNets-style) | Fixed-position solid patch | Localized, spatial |
| Blend Backdoor | Full-image pattern blend | Global, distributed |

## Evaluation Tiers

| Tier | Attacker Knowledge | Description |
|------|-------------------|-------------|
| A (Gray-box) | Knows SSL method, not defense | Transfer attacks |
| B (White-box) | Knows training setup | Stronger attack tuning |
| C (Adaptive) | Knows PRCL details | Discussion + limited experiment |

## What PRCL Claims

PRCL provides **robustness improvements under specific contamination and attack settings**:
- Reduces ASR at low-to-moderate poison ratios (ρ ≤ 10%)
- Preserves clean downstream accuracy (within ~2-3% of undefended baseline)
- PCF scores provide forensic signal separating clean from poisoned samples

## What PRCL Does NOT Claim

- Universal protection against all possible attacks
- Provable robustness guarantees (partial theoretical analysis only)
- Effectiveness against adaptive attackers who specifically target PCF scoring
- Protection at high contamination ratios (ρ > 10%)
