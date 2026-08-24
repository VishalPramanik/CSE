# Method-to-Code Alignment

This repository keeps the implementation close to the paper's mathematical definition rather than embedding experiment-specific shortcuts.

## Core equations

| Paper component | Implementation |
|---|---|
| Joint feature mean and standard deviation, Eqs. (1)-(3) | `cse.method.ContrastiveSubnetErasure.fit_layer` |
| Target/background standardized second moments, Eq. (4) | `sigma_t`, `sigma_b` in `fit_layer` |
| Contrastive Rayleigh quotient / generalized eigenproblem, Eqs. (5)-(6) | `scipy.linalg.eigh(sigma_t, sigma_b_reg)` |
| Eigenvalue-weighted channel salience, Eq. (7) | `salience = sum_j rho_j * v_j[c]^2` |
| Minimal coverage subnet, Eq. (8) | descending salience + cumulative threshold |
| Calibrated attenuation, Eq. (9) | selected-channel `attenuation` |
| Diagonal attenuation and original-space affine, Eqs. (10)-(12) | `scale=1-beta`, `bias=beta*mean` |
| Complete block-output application, Eqs. (13)-(14) | `cse.models.ChannelAffine` / `BlockWithAffine` |
| Optional affine fold-in | `cse.fold` for directly adjacent `Linear` / exact-safe `Conv2d` cases |

## Main-paper hyperparameters

The default configuration is deliberately taken from the main experimental text:

- `alpha = 0.01`
- `k_max = 50`
- `eigen_fraction = 0.30`
- `coverage = 0.85`
- `tau0 = 0.10`
- `lambda0 = 0.50`
- `epsilon = 1e-6`
- non-target set: `10%` per semantically related class

These are centralized in `configs/default.yaml` and `cse/config.py`.

## Resolved manuscript ambiguities

1. **Selected channels vs. all channels.** Sec. 3.4 says attenuation is computed for each *selected* channel, while the supplementary pseudocode writes the transfer function "for all c". The implementation follows the main method description and sets attenuation to zero outside the selected compact subnet. This is required for the claimed localized edit.

2. **Covariance convention.** Eq. (4) explicitly uses `1/n sum h h^T` after joint standardization. The implementation follows that equation rather than silently re-centering target and background features separately.

3. **Eigenvector normalization.** Appendix B.2 states that Euclidean-normalized eigenvectors are used for salience. The generalized eigenvectors are therefore re-normalized to unit L2 norm before Eq. (7).

4. **Block-output placement.** The main method requires attenuation after the complete residual/transformer block so that a residual path cannot bypass the edit. The default implementation wraps complete block/stage outputs. The optional fold utilities are intentionally conservative: exact zero-overhead folding is only performed when the neighboring architecture permits an algebraically exact rewrite.

5. **Dataset split wording.** The main text contains a `40K/10K` CIFAR statement, while the supplementary setup and canonical CIFAR datasets use `50K` train and `10K` test. Dataset loaders use the canonical torchvision splits and do not create an undocumented extra 10K holdout.

6. **MIA balancing.** The appendix asks for all target-train members and an equally sized target-test non-member subset, but CIFAR target-train is larger than target-test. `loss_threshold_mia` builds the required balanced pool using the largest feasible equal-size subset, `min(n_member, n_nonmember)`, before the 50/50 threshold/held-out split.

These choices are documented rather than hidden so that later paper revisions can be reflected by changing one clearly identified implementation point.
