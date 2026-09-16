# DaT Parkinson's Challenge: research and prioritized experiments

Research date: September 16, 2026. Prepared after reading the project log (`docs/project-log.md`), reviewing the training, preprocessing, stacking, assembly and inference code, and inspecting the code/configuration inside `submission_v6.zip`. No competition scans, labels, per-patient features, embeddings or predictions were opened for this review. No training or submission was performed.

The target is lower **private-test log loss for normal versus abnormal examinations**, not clinical Parkinson's diagnosis, accuracy at a threshold, or the highest AUROC in a different dataset. There is no evidence that any proposed experiment will deliver first place. There is also insufficient evidence for the project record's conclusion that the current pipeline has reached the information ceiling.

## Decision for the remaining competition time

The verified deadline is **September 16, 23:59 UTC = September 17, 05:59 Bangladesh time**. At the clock check during this review, it was September 16, 22:15 Bangladesh time, leaving about 7 hours 44 minutes. These are timestamped planning figures, not a live countdown. The organizer says submissions stop at the deadline. [Competition deadline](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/), [post-deadline submissions ruling](https://community.drivendata.org/t/submissions-after-competition-closing/11515).

The user can accommodate at most four smoke tests and four scored submissions and wants to submit only for valuable gains. Treat that as a ceiling, not a target. The actual account/platform quota still applies. Training hardware was not specified, so the time estimates below are planning categories rather than GPU benchmarks.

My first choices are:

1. Reconcile the actual v6 ensemble with the OOF ensemble used to fit its stacker.
2. Evaluate probability averaging and a tightly constrained calibration comparison on valid held-out predictions.
3. Train a version whose second normalization is calculated before background centering.
4. Screen frozen DINOv2 features as a complementary expert.
5. If time permits, train a small slice-sequence CNN with attention pooling.

A validation audit must accompany those experiments. A correct implementation or an interesting paper is not by itself evidence to spend a submission.

The organizer confirms that **the best private-scoring submission counts automatically**, with no explicit selection. Existing submissions therefore remain useful. The organizer does not disclose public/private test sizes; the approximate 320-case public set in the project log (`docs/project-log.md`) must not be treated as known. [Selection and test-size ruling](https://community.drivendata.org/t/test-set-size-and-final-submission-selection/11493).

## What the existing evidence establishes—and what it does not

The project record reports v6 at 0.2403 public log loss and the supplied leaderboard leader at 0.2154. That gap is 0.0249, about a 10.4% reduction relative to v6. I could access the public leaderboard URL but not retrieve its participant rows, so those rankings are the user's snapshot, not an independently refreshed leaderboard.

Registration/VOI features, strong augmentation, and dual normalization are supported by your experiment history. More bagging and routine backbone swaps have diminishing returns. That is a good reason to prioritize changes that affect representation or generalization.

Several conclusions in the project log (`docs/project-log.md`) need qualification:

| Recorded conclusion | More defensible interpretation |
|---|---|
| One temperature/floor submission refuted calibration | It refuted that particular combined adjustment on that public set. It did not isolate temperature from flooring, test an intercept, or establish that all calibration is ineffective. |
| Similar AUROC with better log loss means the gap is not calibration | AUROC does not determine probability calibration. Even identical rankings can produce very different log losses. Similar AUC also does not imply identical errors. |
| Uptake/label contradictions are unfixable by any image model | Contradiction with chosen features or current models is not proof of incorrect labels or absent image information. Independent expert review would be needed to substantiate a noise claim. |
| Every team's LB is worse than CV | Several teams have reported this; it is not evidence about every team. [Public CV/LB discussion](https://community.drivendata.org/t/what-correlation-cv-vs-lb/11507). |
| A roughly 0.004 seed variation makes every smaller result meaningless | Compare paired changes with identical folds and seeds. Paired uncertainty can be smaller than the variation between unrelated runs. |
| The leaderboard gap is approximately one standard deviation | This depends on an undisclosed test size, an assumed distribution, and unknown correlation between teams' errors. It cannot establish statistical equivalence. |

## Confirmed implementation findings

### 1. v6 weights training runs differently in OOF and inference

`assemble_v6.py:52` averages predictions from `v6_dual10` and `v6_dual5b` equally. The former has 10 folds × 3 seeds and the latter 5 folds × 3 seeds. `solution/main.py:131` averages all checkpoint files equally. I confirmed this behavior in `submission_v6.zip`, which contains 45 dual checkpoints.

Thus the intended OOF combination is:

`0.5 × dual10 + 0.5 × dual5b`

but deployed inference uses:

`(2/3) × dual10 + (1/3) × dual5b`.

The tabular combination also deserves reconciliation: assembly averages the base aggregate and new runs equally, whereas inference averages individual model files. The base aggregate may itself contain multiple runs. Preserve run/recipe identity and specify the intended weight hierarchy explicitly.

This is a confirmed disagreement between evaluated and deployed ensemble definitions. It is **not** a measured explanation for the 0.0249 leaderboard gap. Compare both definitions fairly before deciding whether the equal-run or equal-checkpoint version is better.

There is also a dormant issue: assembly writes `group_weights`, but inference ignores it and takes the mean of group members. The shipped v6 config has weights 1:1:1, so this particular omission does not change v6's output. It would break a future nonuniform group-weight experiment.

The existing `parity_v3.py` copies the inference averaging rule. It can verify preprocessing consistency while missing disagreement with the OOF assembly recipe. Add an independent check of the expected weight assigned to each run and checkpoint.

### 2. The normal template is fitted outside validation folds

`solution/train.py:201` builds the template from `crops[(y == 0) & good]` before the per-fold model loop. Cached registration is also shared. Consequently, held-out images and their normal labels influence the template and derived VOIs.

That contaminates the estimate of generalization, even if the template is smooth and any effect is small. All-training-data template fitting is appropriate for the final fit; it is not appropriate for assessing an outer held-out fold. Fit template, atlas and other learned preprocessing only on each outer training partition, and transform its validation partition with those frozen artifacts.

This audit may worsen the reported CV score without improving the model. Its purpose is to make subsequent decisions trustworthy. No experiment in this review measures how large the contamination is.

### 3. The stacker and scanner proxy are not a fully nested pipeline evaluation

`solution/train.py:285` cross-validates the logistic layer over already generated OOF features. That is a useful second-level check, but it does not retrain the complete pipeline inside each outer split. Base models that generate meta-training features can have trained on labels from the nominal outer validation fold. Hyperparameter and clipping selection then reuse the reported validation predictions.

`proxy_eval.py:16` fits the stacker to random-fold OOF features and labels from all subjects, then scores scanner-held-out features of those same subjects. This is a distribution-shift diagnostic, not a clean unseen-subject validation of the full stack.

For a strict estimate, outer validation labels must be excluded from every fitted stage, including the meta-model, preprocessing, feature selection, calibration and weighting. Generate inner OOF features within the outer training set, fit the stacker there, and predict outer validation with base models that never saw it. Existing OOF arrays can screen candidates but cannot retroactively become fully nested by changing their fold column.

### 4. The dual channel is normalized after centering

The training and inference call `prep_input(x) = (x - 1) / 2` before `Net2D.forward`. The dual branch divides that centered image by its own p99. If `I` is intensity, `B` is the estimated background and `Q` its intensity quantile, the second branch is effectively:

`(I - B) / (Q - B)`

rather than `I / Q`, apart from clipping, interpolation and augmentation effects. The former still depends on `B`, contrary to the comment that it is independent of the background estimate.

This need not be a bad representation—the recorded experiment improved—but it is not the background-independent alternative its name suggests. Test true raw-positive percentile normalization as a new training experiment. Do not change this transform under existing weights.

### 5. Physical geometry handling needs an obliquity audit

`load_canonical` returns data and voxel spacings, not the full affine. `preprocess` resamples using a diagonal scale. NIfTI canonicalization permutes/flips axes; it need not remove oblique rotations or shear. Thus this is not a complete voxel-to-world resampling implementation for arbitrary NIfTI geometry.

The later registration may compensate for many rotations. There is no evidence from this review that problematic affines occur or that this causes a scoring loss. Check locally, and distinguish a header-affine issue from simulated rotation of an already resampled image. A proper alternative uses the full affine and composes transforms into a single final interpolation.

## Ranked experiment inventory

These are experiment proposals, not measured improvements. Priority is specific to the existing solution, available time and four-submission ceiling. Screening cost excludes full clean nested validation, repeated seeds and packaging.

| ID | Experiment | Priority now | Initial cost | Why it is not simply a repeat |
|---|---|---|---|---|
| A01 | Explicit run weights and ensemble parity | First | CPU / existing outputs | Confirmed v6 discrepancy |
| A02 | Clean validation and deployment-like prediction aggregation | First | From CPU audit to retraining | Existing nested/proxy estimates have limitations |
| A03 | Probability pooling versus logit pooling | High | CPU if member predictions exist | Changes ensemble behavior on disagreements |
| A04 | Constrained affine/beta calibration | High, gated | CPU | One temperature-plus-floor test is insufficient |
| A05 | True raw-positive dual normalization | High | One CNN recipe | Current dual still depends on background |
| A06 | Frozen DINOv2 features and small head | High | Feature pass + CPU fitting | Different representation, no CNN retraining initially |
| A07 | Multi-reference normalization and ratio features | Medium-high | CPU preprocessing + tabular CV | Tests denominator robustness directly |
| A08 | 2.5D slice sequence + attention | Medium-high | CNN training | Retains slices rather than averaging them away |
| A09 | Joint image/VOI feature fusion | Medium-high | CNN training | Existing stack only sees prediction summaries |
| A10 | Regional auxiliary supervision | Medium | CNN training | Richer supervision than binary label/noisy-OR |
| A11 | Acquisition-focused augmentation and consistency | Medium | CNN training | Broader nuisance model than isotropic blur/noise |
| A12 | Freeze BatchNorm / alternative normalization | Medium | CNN training | Addresses batch-correlated augmentation |
| A13 | SAM optimization / within-run SWA or SWAD | Medium | Retraining | Across-seed soup failure does not test this |
| A14 | ROI-pooled or pretrained small 3D expert | Medium, time permitting | Training + license check | Existing 3D model is scratch-trained, globally pooled |
| A15 | Geometry-preserving preprocessing diversity | Medium | New preprocessing + training | Tests representation bias, not extra yaw TTA |
| A16 | Learned bilateral comparison | Medium | CNN training | Noisy-OR imposes a specific decision rule |
| A17 | Partial-volume-aware integrated uptake | Medium | CPU features + CV | Mechanistic alternative to blur harmonization |
| A18 | Training-domain balancing / Group DRO | Medium-low today | Retraining | Changes training objective, not scanner stack features |
| A19 | MixStyle / CORAL / Fishr | Medium-low today | Retraining | Learns resistance to training-domain variation |
| A20 | Self-supervised local pretraining | Longer project | Substantial training | Uses training images without extra labels |
| A21 | Segmentation pretraining / local expert ROI annotation | Longer project | Annotation + training | Adds anatomy supervision |
| A22 | Focused deep-radiomic residual expert | Conditional | CPU / feature extraction | Target residual errors, not indiscriminate features |
| A23 | Learned extrastriatal branch | Conditional | CNN training | Generic context statistics did not test it |
| A24 | Uncertainty-conditioned shrinkage / residual expert | Conditional | CPU to small model | Probability adjustment conditioned on evidence |
| A25 | Conservative noisy-label learning / distillation | Conditional | Review + training | Different from unconditional smoothing or deletion |
| A26 | External data and pretrained asset search | Low today | Variable | Eligible raw DaT data remains unverified |

### A01. Make ensemble weights explicit

Represent the ensemble as recipe → split → seed → fold → augmentation. For each recipe, average folds within a seed, seeds within a split, then apply the same explicit split and recipe weights used in OOF construction. Do the equivalent at validation time using only genuinely held-out predictions.

Compare the current deployed weighting, equal-run weighting, and at most one regularized alternative. Fit any stack/calibration on features constructed with that same rule. A hierarchical weighting fix can be evaluated without retraining all networks if suitable independent predictions are already available; otherwise obtain a small clean holdout benchmark.

Also check model/config pairing and explicit feature order. Assertions about UIDs and fold membership should run locally, with only pass/fail returned. Do not estimate improvement by evaluating every final ensemble member on training cases it saw.

### A02. Repair validation and measure the quantity that matters

Keep two evaluations: stratified patient-level CV for the likely mixed-center task, and held-out acquisition-family CV as a robustness stress test. Voxel size is a proxy for acquisition, not a hospital identifier. Add dimensions, spacing on all axes and non-pathology acquisition descriptors only where they support stable grouping; do not claim to know the center from them.

Fit all train-derived artifacts within the appropriate partition. When evaluating an ensemble, its held-out predictions should approximate the training size and aggregation used at inference. Ordinary OOF gives each subject predictions from one held-out fold per split, whereas deployment averages many overlapping models; their variance and calibration can differ.

Use paired per-examination loss differences internally, with patient resampling and acquisition-group resampling as separate sensitivity analyses. Return whole-set scalar summaries to the assistant. Because the same folds have already guided many experiments, confidence intervals from them are descriptive, not protection from adaptive model selection. Reserve a clean confirmation split if practical.

### A03. Probability averaging and conservative ensemble selection

The current pipeline averages logits. Compare:

`p_probability = sum(w_m * sigmoid(z_m))`

with:

`p_logit = sigmoid(sum(w_m * z_m))`.

Probability averaging tends to preserve disagreement rather than allowing a very large logit to dominate. It is not guaranteed to beat logit pooling. For log loss, convexity guarantees a fixed probability mixture is no worse than the weighted average of its members' losses on the same cases; it does not guarantee beating the best member.

Start with a few family-level components rather than hundreds of free checkpoint weights. Fit nonnegative weights summing to one, with shrinkage toward equal weights, using inner-held-out data. Compare to the existing logistic stack and a small fixed blend. Evaluate marginal ensemble gain, not just standalone AUROC or raw prediction correlation. Keep a weaker standalone model only if its inclusion improves held-out loss reliably.

### A04. Revisit calibration carefully

Your stacker already provides a form of calibration, so expect diminishing returns. Compare identity, temperature, affine-logit scaling and monotone beta calibration using the exact final ensemble output. For affine calibration use `sigmoid(a * logit(p) + b)`, with `a > 0`; unlike temperature alone, this can adjust the overall intercept. For beta calibration use `sigmoid(a*log(p) - b*log(1-p) + c)`, constrain `a,b >= 0`, and regularize toward identity.

Use inner calibration/selection and untouched outer scoring. Investigate classwise losses and reliability internally; do not use assumed public-test prevalence. Fit the intercept only from permitted training partitions. Start without an added probability floor and evaluate any floor separately. A small hand-picked temperature and a floor changing together cannot isolate either effect.

The general methods are supported by [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html) and [Kull et al., beta calibration](https://proceedings.mlr.press/v54/kull17a.html). Their papers do not predict a gain on this competition.

### A05. True background-independent normalization

Use the successful EfficientNet-B0 recipe, existing registration and strong augmentation as the control. From a positive image `U`, construct one branch `U = I/B` and another `U / quantile(U, 0.99)`, applying centering/scaling only afterwards. Clip with consistent rules and a denominator guard. The second branch is approximately invariant to `B` unless prior clipping has already discarded information.

A minimal experiment keeps six channels and changes only the order of operations. A more expressive variant uses separate encoders or a shared encoder with a learned branch indicator. Another uses L2 normalization over a striatal-plus-reference region. Preserve the amplitude-bearing branch: a shape-only representation can discard useful evidence of bilateral reduction.

[Zhou and Tagare](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006242/) analyze multiplicative normalization and test unit-sphere self-normalization. Their method is not simply p99 division, and their PPMI results are methodological evidence rather than eligible weights. Judge this experiment by incremental calibrated ensemble loss over the current dual model, using matched seeds.

### A06. Frozen DINOv2 as a complementary expert

Use a standard DINOv2 ViT-S/14 or ViT-B/14 checkpoint with its specific model card and license saved. Begin with fixed grayscale-replicated images or three adjacent physical slices; use both a striatal crop and a slightly wider crop. Extract CLS and region-pooled patch features. Fit a strongly regularized logistic head; fit any dimension reduction inside training folds. Frozen public features can be cached once, while learned transformations and label-dependent preprocessing remain fold-specific.

First ask whether a small blend improves the existing ensemble. If it does, unfreeze only the last blocks or train adapters with a lower backbone learning rate. Small patches and striatal pooling are useful because the target occupies a limited area. Natural-image pretraining may still fail on SPECT; this is a cheap representation screen, not a guaranteed foundation-model advantage.

The standard [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) specifies Apache 2.0 and supports linear downstream heads. Do not generalize that license to every asset in the repository: [XRay-DINO weights](https://github.com/facebookresearch/dinov2/blob/main/README.md) have a noncommercial license. DINOv3 is explicitly disfavored for prize eligibility by the [organizer ruling](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488).

### A07. Multi-reference normalization

The current denominator is a trimmed head-mask mean around the striatum. Compare anatomically defined occipital reference, an extrastriatal brain reference with exclusions, and the existing denominator. Compute SBRs and ratios with each; train a small tabular model or feed separate normalized views into a CNN. Reference-region definitions must be robust to cropped FOV and registration failure.

Start with existing voxel data and a few high-value ratios: worse-side posterior putamen/reference, putamen/caudate, bilateral mean/reference and reference-to-reference ratios. Test whether adding these improves errors that persist with current features. Do not append hundreds of nearly duplicate ratios without regularization.

The [clinical uncertainty paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/) describes an extrastriatal p75 reference with explicit exclusions. Use it as an alternative design, not as evidence that the same denominator is optimal here. The self-normalization paper above supplies the reason to test denominator sensitivity.

### A08. Slice-sequence learning instead of three slab averages

Extract approximately 9–17 slices spanning the striatum, in physical coordinates, and encode each slice with a shared small ImageNet CNN. Aggregate embeddings with a short bidirectional GRU/LSTM or a shallow transformer, followed by gated attention. Supply relative z position and a validity mask. Begin with registered scans, consistent in-plane crops and modest slice dropout.

This preserves variation along the superior–inferior axis and allows the model to discount irrelevant slices. It differs from both the failed scratch 3D CNN and the hemisphere noisy-OR model. First compare attention pooling to simple mean pooling so that extra parameters must justify themselves.

The [RSNA 2024 winning solution presentation by NANACHI](https://speakerdeck.com/nanachi/rsna2024zhen-rifan-ri) describes improvement from slice encodings, attention pooling and then a bidirectional LSTM. The [Kaggle writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution) is linked by RSNA but its body was not retrievable in this review. Transfer the sequence-learning principle; MRI competition scores do not establish SPECT performance. [Attention-based MIL](https://arxiv.org/abs/1802.04712) provides the generic pooling construction.

### A09. Joint image–quantification fusion

The current stack compresses each model family to one prediction. That prevents interaction between an image feature and a specific quantitative measurement before the final decision. Try a CNN embedding plus an MLP on a small VOI feature set, concatenated into a regularized binary head. Add separate image-only and VOI-only auxiliary heads or modality dropout to discourage dependence on one branch.

A second variant pools CNN feature maps over caudate/putamen masks and fuses those embeddings with corresponding uptake ratios. Start with concatenation; only try cross-attention if it earns a gain. Every mask/template and feature scaler is fit within the training fold.

[Durand de Gevigney et al., January 2026](https://link.springer.com/article/10.1007/s10278-025-01831-w), combine image and scalar encoders with transformer fusion. The accessible abstract reports heterogeneous cohorts and transfer learning, but includes age and PPMI data. This competition adaptation would omit age and train on eligible data. The full article was subscription-restricted; the [ECR 2026 poster](https://epos.myesr.org/poster/esr/ecr2026/C-21138/conclusion) is additional author-provided evidence, not an identical experiment.

### A10. Regional auxiliary supervision trained locally

Predict continuous regional uptake or ordinal severity for left/right caudate and putamen alongside the scan label. A starting objective is binary cross-entropy plus a low-weight Huber loss on standardized regional ratios. Alternatively fit train-only regional mixtures, turn their ordered categories into weak auxiliary targets, and use ordinal heads. The supplied binary expert label remains the main objective.

This could give the encoder a structured learning task and distinguish symmetric reduction, selective putaminal loss and atypical patterns. It does not manufacture new ground truth: VOI-derived targets can propagate VOI mistakes, so keep auxiliary weights small and test ablation against the same binary model.

[Buddenkotte et al., 2025](https://link.springer.com/article/10.1007/s12149-025-02038-3) demonstrate five-category regional prediction. **Do not use its released regional checkpoints for the prize submission:** its training split includes PPMI, as specified in the dataset/training sections, and the [competition ruling excludes PPMI-trained models](https://community.drivendata.org/t/is-ppmi-dua-gated-acceptable-as-external-data-for-prize-eligibility/11472). This matters even though the shared repository is MIT-licensed. The architecture/objective can be reimplemented and trained locally.

### A11. Extend acquisition augmentation, with consistency checks

Keep the known-effective strong augmentation as the control. Test one addition at a time: anisotropic blur, resample/downsample/upsample to varied spacings, spatially correlated noise, smooth multiplicative fields, or mild shift-mixtures mimicking motion. These are approximations in reconstructed-image space; Poisson noise on a rescaled reconstruction is not equivalent to simulating photon acquisition.

Apply geometric transforms consistently across all channels and masks. Compare per-sample blur strengths to the current one-blur-per-batch sampling. Add a small consistency penalty between weakly and strongly perturbed versions of the same training scan only if the perturbation preserves its label.

[Buddenkotte and Buchert, 2024](https://jnm.snmjournals.org/content/65/9/1463) directly support blur/noise augmentation for cross-camera DaT robustness, which your experiments already corroborate. The additional perturbations above are proposals. Large perturbations can erase a subtle putaminal deficit; validation must check their effect on the task, not only visual realism.

### A12. BatchNorm and pretrained-feature preservation

Your augmentation chooses whether to blur and its blur strength for an entire batch. BatchNorm statistics can therefore track unusually homogeneous synthetic batches. Compare frozen BatchNorm running statistics with trainable affine parameters against ordinary training. For a small new 3D model, compare GroupNorm with BatchNorm. Keep all inference models in evaluation mode.

Do not indiscriminately replace norms in a pretrained backbone: that alters its learned representation. Also note that InstanceNorm can suppress useful absolute intensity. A normalization change needs the amplitude branch and calibration re-evaluated. This is a code-motivated hypothesis; no local evidence currently shows BatchNorm is responsible for the plateau.

### A13. SAM and averaging within a training trajectory

Try SAM on the strongest small CNN, initially with the same data, epochs and augmentation. It increases training cost because the update typically requires two forward/backward passes. Keep augmentation samples consistent between the two passes and handle BatchNorm buffers carefully. Evaluate a small parameter set rather than sweeping broadly.

Alternatively, average nearby late checkpoints from one training run in a stable validation window. The project already uses EMA, so ordinary averaging may add little. SWAD's window selection is a different test from averaging unrelated seeds. Recompute any needed running statistics from training data only.

[SAM](https://arxiv.org/abs/2010.01412) and [SWAD](https://arxiv.org/abs/2102.08604) motivate flatter solutions/generalization. A failed across-seed model soup does not refute either. Neither supplies a measured DaT gain.

### A14. A different kind of 3D expert

The existing scratch 3D model downsamples repeatedly and globally averages its features. Test a shallow residual model with less early downsampling and ROI-specific pooling over left/right caudate/putamen. Compare a learned axial aggregation or separable 2D-plus-depth convolution before investing in a large volumetric transformer.

If using pretrained 3D encoders, [MedicalNet](https://github.com/Tencent/MedicalNet) provides medical 3D ResNet initialization and states an MIT license. Verify the exact checkpoint and terms, and port its architecture to the current runtime; its historical environment instructions are not the competition runtime. Training a small [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) encoder is another longer experiment, not a reason to assume segmentation weights transfer to DaT classification.

Retain only if it improves an ensemble, particularly on held-out acquisitions. The earlier poor 3D experiment makes this lower priority than corrected normalization or 2.5D sequence learning.

### A15. Preserve morphology through preprocessing diversity

Compare the current normal-template affine registration with rigid-only registration and an unregistered but accurately localized branch. Affine scaling can improve correspondence while changing dimensions that carry morphology. High NCC does not prove that every diagnostically useful relation was preserved.

Use the same model and splits to isolate the representation change. A small fixed blend of rigid and affine experts may be more useful than several similar backbones. For the geometry audit, compose the original NIfTI affine and final transform and interpolate once where possible.

Multi-template selection based solely on each scan is not automatically leakage merely because the chosen template correlates with pathology; it can be a label-dependent representation. The prior measured negative result is still a reason to deprioritize it. Any templates learned from labels must be fold-specific, and any choice using a validation/test label would be invalid.

### A16. Learn the bilateral relationship

Mirror the right striatum into a shared anatomical frame and use a shared encoder for each side. Feed `h_left + h_right`, `abs(h_left - h_right)` and bilateral quantitative features to a small classifier. This is invariant to exchanging left/right while retaining evidence of symmetric loss and asymmetric shape.

The previous noisy-OR model assumes scan abnormality can be inferred from either independently scored hemisphere. That assumption is restrictive; the learned comparison tests a different hypothesis. Keep enough context to distinguish putamen from caudate and avoid independently rescaling each hemisphere so aggressively that severity differences vanish. Whole-brain or wider-crop evidence can join as a small third input.

### A17. Integrated uptake and partial-volume-aware features

The SPECT spatial resolution and small striatal structures make partial-volume effects important. Instead of harmonizing every image to the same blur, compute integrated excess counts within a larger fixed striatal VOI, as well as more local ratios, and retain both. An experimental integrated feature is `voxel_volume × sum(I/B - 1)` over a predefined VOI; do not call it a validated clinical SBR without the appropriate volume assumptions.

Test several prespecified VOI dilations, worst-side and bilateral summaries, and stability under blur. A more advanced version fits a low-dimensional regional source model convolved with a few candidate point-spread functions and marginalizes over them. Resolution and uptake can be confounded, so constrain the model and reject unstable corrections.

The [Southampton-method discussion and comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC9278684/) and [scanner/reconstruction study](https://link.springer.com/article/10.1186/s13550-016-0253-0) motivate this route. It differs from globally blurring all scans, which your history found harmful.

### A18. Acquisition-balanced training and Group DRO

First compare ordinary sampling with capped inverse-frequency weights for acquisition families, retaining label coverage and avoiding domination by tiny groups. Any reweighting changes the effective training distribution, so score and calibrate on natural-prevalence held-out data.

If reliable groups exist, test a mixture of average cross-entropy and group-robust loss. Prefer a bounded objective to pure minimization of the single worst tiny group. Use actual center identifiers only if legitimately supplied, not inferred as facts from voxel size.

[Group DRO](https://arxiv.org/abs/1911.08731) emphasizes that strong regularization matters for worst-group generalization. [DomainBed](https://arxiv.org/abs/2007.01434) warns that carefully tuned ordinary training is a strong baseline. This is appropriate only if robustness gains survive normal CV; the private set may mix familiar centers rather than contain unseen ones.

### A19. MixStyle, CORAL and Fishr

Try MixStyle in early feature blocks with a low probability; train on multiple acquisition families and mix style statistics within permitted training batches. Keep an unmodified intensity branch because intensity is partly diagnostic here. CORAL aligns training-domain feature covariances; Fishr matches gradient-variance information across training domains. Both are alternatives, not additions to stack simultaneously.

These methods can suppress scanner shortcuts, but also erase biology when disease prevalence differs between acquisition groups. Start with a small single-method ablation and select by held-out-acquisition and ordinary log loss. Sources: [MixStyle paper](https://arxiv.org/abs/2104.02008), [official implementation](https://github.com/KaiyangZhou/mixstyle-release), [Fishr](https://arxiv.org/abs/2109.02934), [DomainBed implementations](https://github.com/facebookresearch/DomainBed).

Do not run target/test-distribution alignment or estimate covariance across the test set.

### A20. Local self-supervised learning

Pretrain an encoder on two perturbations of the same permitted training scan using VICReg or a related consistency objective. Both views should retain the striatum. Preserve discriminative gradients by avoiding augmentations that force a clear abnormal region to match a crop containing only background.

An alternative masks patches of a small 3D crop and reconstructs latent features or intensities, then fine-tunes on the binary label. Treat scanner recognition as a potential shortcut; test embeddings with acquisition-held-out splits. For strict inductive CV, pretrain only on the outer training partition. The final fit can use all competition training images.

[VICReg](https://arxiv.org/abs/2105.04906) and [VICRegL](https://arxiv.org/abs/2210.01571) provide global/local representation objectives. This is a longer project with uncertain return from a relatively small dataset. It is lower priority today than starting from a permitted frozen pretrained encoder. Never pretrain on the hidden test set.

### A21. Anatomy-supervised pretraining and local expert annotation

Use train-only weak masks, or new local expert annotations of training images, to supervise localization/segmentation of striatal regions. Then classify from ROI-pooled encoder features and a global feature. Preserve annotations and instructions needed for reproducibility. Annotation must remain local and may not involve hidden test cases.

The [RSNA 2025 first-place author writeup](https://github.com/uchiyama33/rsna2025_1st_place/blob/main/solution_v1.md) describes segmentation-derived representations, regional pooling and auxiliary supervision. Its vessel problem is different; its annotations and weights cannot be assumed eligible or helpful here.

This approach is useful if localization or region definition is a verified bottleneck. Given the existing stress tests and time remaining, expert annotation plus model development is unlikely to be today's best use of resources.

### A22. A focused deep-radiomic residual model

Extract a compact set of new features with a specific mechanistic role: threshold-versus-volume curves, posterior tail persistence, integrated excess counts, surface curvature, or region-pooled deep embeddings. Assess stability to small resampling/blur changes. Fit an elastic-net logistic model or small boosted model within folds and test its marginal benefit beside v6.

Do not indiscriminately add texture families to the existing 173 features. At SPECT resolution, many texture metrics can be reconstruction signatures. The prior failures of PCA, SVM, MLP and extra VOI features lower this experiment's prior probability of success.

[Jiang et al., 2024](https://ejnmmiphys.springeropen.com/articles/10.1186/s40658-024-00651-1) combine deep and radiomic features, but predict Hoehn–Yahr stage/progression and use PPMI. That supports examining complementarity, not expecting their reported results on normal/abnormal competition labels.

### A23. A learned extrastriatal branch

Train a low-capacity wider-brain branch with independently defined reference normalization and fuse it with the striatal branch. Assess whether spatially localized information outside the striatum helps when the striatal evidence is ambiguous. Generic brain-size/ventricle/cortical summary statistics, already unsuccessful, do not test this learned spatial representation.

[Nazari et al., extrastriatal analysis](https://link.springer.com/article/10.1038/s41598-021-02385-x) report higher accuracy with full images than striatum-only images and identify possible extra-striatal contributions. Their internal study is not proof that those signals generalize across French centers. Scanner artifacts and normalization effects are strong competing explanations. Require acquisition-held-out gain; drop this branch if its only advantage is random CV.

### A24. Uncertainty-conditioned probability adjustment

If errors concentrate in model disagreements, use a tiny held-out-trained adjustment based on the ensemble's probability, disagreement and a few per-scan quality indicators. A starting proposal is `p_adjusted = (1-alpha(q))*p + alpha(q)*pi_train`, with a constrained, low-capacity `alpha`. Compare to a constant shrinkage baseline.

Disagreement does not automatically mean the label is unpredictable, and shrinking toward the prior can harm confident correct cases. Any quality/disagreement relationship must be estimated from genuinely held-out predictions. Do not compute quality thresholds or priors across the test set.

The [DaT uncertainty study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/) investigates identifying uncertain cases through network disagreement. The competition requires a probability for every scan, so convert that idea into a validated probability model rather than clinical abstention. The supplied labels offer no uncertainty annotations.

### A25. Conservative label-noise learning, mixup and distillation

Use out-of-fold disagreement only to prioritize local human review, not to declare labels wrong. Preserve the official labels for scoring. A low-weight auxiliary target from a teacher, generated without training on that subject, can regularize a student; always compare against training on original labels alone. A teacher does not add independent information and can reproduce the same errors.

If mixup has not had a proper paired experiment, try a small alpha such as 0.1 or 0.2 and turn it off late in training. The code supports mixup, but support is not evidence it was evaluated. Interpolated scans/labels are regularization examples, not clinical simulations. Focal loss, aggressive co-teaching, hard-example mining and generalized cross-entropy are lower priority because the objective is well-calibrated log loss and genuinely hard cases may be crucial.

[Confident learning](https://arxiv.org/abs/1911.00068) offers methods for estimating label issues under assumptions; it does not prove this dataset's labels are wrong. [Mixup](https://arxiv.org/abs/1710.09412) motivates regularization between training examples. Any student influenced by excluded PPMI-derived checkpoints remains a problematic prize candidate; distillation is not a licensing workaround.

### A26. External data and model assets

External pretraining can be valuable, but I did not verify a sufficiently large, freely redistributable, commercially usable raw DaT dataset that solves today's data problem. Kaggle, Zenodo and general-web searches produced PPMI references, task-mismatched resources and releases without raw SPECT volumes. This is a search result, not proof that no eligible dataset exists.

Use standard permitted pretrained image encoders first. For each asset save the exact source, revision, checkpoint hash, license and any applicable organizer ruling. The [ImageNet clarification](https://community.drivendata.org/t/clarification-on-imagenet-pre-trained-weights-under-open-source-commercial-rules/11475) focuses on permissive terms for the model/weights; this is not permission to ignore an explicit PPMI ruling.

| Resource | Practical verdict |
|---|---|
| Existing timm ImageNet backbones | Keep; retain license evidence for exact weights. |
| Standard DINOv2 | Plausible permitted candidate; standard model card says Apache 2.0. |
| DINOv3 | Exclude from prize shortlist under organizer guidance. |
| XRay-DINO | Exclude: noncommercial weights, and X-ray/SPECT mismatch. |
| PPMI scans or PPMI-trained models, including the 2025 regional DaT checkpoints | Exclude from prize shortlist under the explicit PPMI ruling. |
| ThomasBudd binary uncertainty/robustness weights | Already tested without useful gain. Audit each checkpoint separately; sharing a repository with the regional model does not establish identical training provenance. |
| MedicalNet | Conditional: inspect exact released weights, license and runtime compatibility. |
| Fast DaTscan restoration models | Low priority: low-count restoration is a different task; code license alone does not settle all asset rights. [Author repository](https://github.com/YazdanSalimi/Fast-DatScan-SPECT-Imaging). |
| Kaggle repackaged DaT images | A Kaggle license badge does not override original-source restrictions or establish patient independence. |
| Generic MRI/PET data or atlases | Only if task-relevant and licenses verified; no assumption that freely downloadable means commercially usable. |

## What the clinical literature changes about the search

The strongest lesson is to distinguish image interpretation from disease diagnosis and stage prediction. Many seemingly spectacular results solve a cleaner or different problem.

| Resource checked | Relevance and limit |
|---|---|
| [Booij et al., 1998](https://jnm.snmjournals.org/content/jnumed/39/11/1879.full.pdf) | Early ROI/VOI reproducibility work; supports quantitative measurements, not a modern model ranking. |
| [EANM/SNMMI dopaminergic imaging guideline](https://pmc.ncbi.nlm.nih.gov/articles/7300075/) | Morphology, reference measurements, acquisition variation and partial-volume effects guide representation design. SPECT resolution is much lower than the 2 mm resampling grid. |
| [Nazari et al., explainable classification](https://link.springer.com/article/10.1007/s00259-021-05569-9) | Close task: visually classified clinical DaT scans; also shows selected atypical structural/vascular cases were excluded. Its performance cannot be directly transferred to this dataset. |
| [Quan et al., 2019](https://arxiv.org/abs/1909.04142) | PPMI-based DaT CNN reference; useful historically, no compelling untried recipe beyond current work. |
| [Zhao et al., 2022](https://link.springer.com/article/10.1007/s00259-022-05804-x) | Uses [11C]CFT PET and differential parkinsonism diagnosis. It is not a matched [123I]FP-CIT normal/abnormal SPECT benchmark. |
| [2026 systematic review](https://link.springer.com/article/10.1007/s10072-025-08768-6) | Accessible abstract covers 130 studies across six modalities; useful map, not evidence that a single model family wins here. Full text was subscription-restricted. |
| [2024 ComBat DaT study](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1306546/full) | Supports investigating site effects. Standard cohort-fitted harmonization cannot be applied by estimating parameters from hidden test cases. Train-fitted adjustments need a fixed per-case inference path and an unseen-site fallback. |
| [2025 projection-based prediction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154367/) | Raw projection inputs are unavailable here, so its primary method is not implementable for this competition. |

The supplied bibliography has some inconsistent author/page metadata. The report links papers by their resolved title/DOI instead of reproducing those inconsistencies.

## What GitHub and Kaggle contribute

The [aryankathpalia competitor repository](https://github.com/aryankathpalia/dat-spect-parkinsons-competition) documents stronger 2D representation performance than its scratch 3D baseline and a substantial CV/LB gap. It does not provide evidence of a better solution than yours. The [SAIFULLAH repository](https://github.com/SAIFULLAH-SHARAFAT/DaT-Parkinson-Challenger) was accessible; the `guelmbaye/dat-parkinsons` page could not be retrieved during this review. I did not find a verified public first-place DaT solution to copy.

The useful transfer from other competitions is region localization, slice aggregation, auxiliary anatomy supervision, and diverse representations. The [RSNA official 2024 winners page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge) points to primary winner writeups. Use those design ideas without transferring their private-test adaptation, extra-data assumptions, labels, or competition-specific losses.

A model with a worse standalone score can help an ensemble. Conversely, a larger ensemble of very similar models may barely improve performance. Select additions by held-out reduction in the final metric.

## Submission gates: at most four, only if earned

Define the gates before inspecting more experiment results. Suggested thresholds below are practical decision rules for this deadline, not a guarantee of statistical significance or public-LB improvement.

1. **Valid comparison.** Same subjects, identical splits and matched seeds; no subject predicted by a checkpoint that trained on it. Separate cheap OOF screening from clean confirmation.
2. **Meaningful magnitude.** Aim for at least 0.005 absolute improvement in ensemble log loss in clean confirmation. A 0.001 gain after trying many variants is not persuasive here. A model with slightly less average gain but a clear prespecified acquisition-robustness improvement merits individual judgment, not automatic rejection.
3. **Repeatability.** Improvement survives a second split/seed where feasible and is not driven by one record. Use paired resampling internally; an improvement frequency of at least 90% can be a screening rule, but is not a valid post-selection p-value.
4. **Robustness.** Check natural-prevalence CV and held-out-acquisition performance. Investigate a large deterioration in a major acquisition family rather than hiding it in the mean. Only whole-set, non-record-level summaries are sent to the assistant.
5. **Exact deployability.** Frozen preprocessing, weights, model hierarchy, calibration, feature order and probability clipping are identical to the validated definition. Check the archive's contents, not only the working tree.
6. **Runtime margin.** Local runtime test, then smoke test. A smoke pass establishes execution and formatting, not generalization. Budget upload, queue and scoring time, and leave headroom below the official limits.

Potential submission sequence—skip any slot that fails its gates:

| Slot | Candidate | Evidence required |
|---|---|---|
| 1 | Reconciled v6 ensemble, possibly with validated pooling/calibration | Independent estimate of useful gain; a bug fix alone does not qualify |
| 2 | Slot 1 plus genuinely complementary frozen DINOv2 expert | Paired ensemble gain on confirmation data |
| 3 | Best blend incorporating true raw-positive dual normalization or slice sequence | Retrained recipe beats its matched control and improves the ensemble |
| 4 | Final independently supported blend or acquisition-robust variant | Additional evidence beyond tiny adaptive OOF gains; retain as contingency otherwise |

Stop broad experimentation when there is no longer time for verification and submission processing. Four variants of temperature or clipping are not a good use of the remaining slots. The first three research priorities can run as local experiments without committing to any upload.

The [runtime specification](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/) requires Python 3.12, offline inference, a 3-hour full limit and a 6-minute smoke limit on the specified A100 environment. The [runtime dependencies](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime/blob/main/runtime/pyproject.toml) include the principal PyTorch, timm, MONAI, registration and classical-ML libraries needed for the shortlist. Verify locked versions and the built image rather than installing packages at inference.

## Boundaries that directly affect these recommendations

The [organizer's AI-assistant clarification](https://community.drivendata.org/t/can-abstracted-or-encoded-data-be-shared-with-ai-assistants-for-preprocessing-advice/11511) permits general information and whole-training-set aggregates, but excludes individual records and sample-level transformed representations. The project record's older rule of allowing groups of nine or more is not a substitute for that newer clarification. Keep scans, labels, features, embeddings, figures and patient-level predictions local.

Each hidden test case must be processed independently. Do not use test pseudo-labeling, test-set pretraining, target-batch statistics, test-cohort ComBat, inferred test prevalence, test-rank normalization, or batch-dependent model adaptation. Ordinary per-scan normalization and frozen-model augmentation are different: their output must not depend on other test cases. Source: [competition rules](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/).

Methods requiring CT, MRI, raw projections, age, sex, symptoms or medication records are not immediately available from this input specification. Do not invent those inputs or treat inferred demographics as independent information. Attention/saliency plots are diagnostic tools, not evidence of causal validity or automatic leaderboard gain.

Data use after the deadline requires a separate applicable license. Longer-project proposals here are only actionable with authorized data access during the challenge or after an eligible public release. No deletion or external communication was performed in this review.

## Final recommendation

The strongest next step is a reproducible local comparison of the shipped ensemble, the intended ensemble and a minimal alternative pooling/calibration definition. In parallel with that work, the best new representation experiments are true raw-positive normalization, frozen standard DINOv2, and slice-wise feature aggregation. Joint image/VOI fusion and regional auxiliary targets are the most relevant deeper follow-ons.

The evidence supports continuing with a more precise experimental plan. It supports neither a promise of first place nor the claim that the current solution has exhausted what the images can reveal.
