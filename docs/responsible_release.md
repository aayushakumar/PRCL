# Responsible Release

## Defense-First Framing

This repository is designed and released as a **defense research tool**. The primary contribution is PRCL — a method for making contrastive SSL pretraining more robust against data poisoning attacks.

Attack adapters are included **solely for controlled evaluation** of the defense. They implement well-known, previously published attack patterns (BadNets-style patch backdoor, blend backdoor) and do not introduce novel offensive capabilities.

## Safety Mechanisms

### Dual-gated attack execution
Attack code paths require **both**:
1. Environment variable: `PRCL_ALLOW_ATTACKS=1`
2. Config flag: `attack.enabled: true`

If either is missing, attack code refuses to execute with an explicit error message.

### Safe defaults
- `attack.enabled: false` in all default configs
- `defense=none` (no defense overhead by default; opt-in to PRCL)
- No one-click offensive pipeline

### Attack metadata logging
When attacks are enabled:
- Full attack configuration is saved to `attack_metadata.json`
- Poison indices are recorded for forensic evaluation
- All metadata is local to the run directory

## Guidelines for Public Release of Results

If publishing results from this system:
1. Include the threat model and specify attacker capabilities
2. Disclose which safety defaults were modified
3. Report both clean accuracy and ASR (not just defense efficacy)
4. Avoid releasing operational details beyond what is needed for reproducibility
5. Reference upstream attack papers when using their trigger patterns

## Contact

For responsible disclosure of security issues in this codebase, please open a private issue or contact the maintainers directly.
