import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help=
        "Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
        " Note that same model used for all datasets, so much have same classnames"
        "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help=
        "Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=
        ('Interpolation coefficient for ensembling. '
         'Users should specify N-1 values, where N is the number of '
         'models being ensembled. The specified numbers should sum to '
         'less than 1. Note that the order of these values matter, and '
         'should be the same as the order of the classifiers being ensembled.'
         ))
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.")
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.00001,
                        help="Learning rate.")

    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")

    parser.add_argument("--ls",
                        type=float,
                        default=0.0,
                        help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--vis_calibration",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help=
        "Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=
        "Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help=
        "Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )

    parser.add_argument(
        "--ft_data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )

    parser.add_argument('--ce_ablation', action=argparse.BooleanOptionalAction)


    parser.add_argument("--dataset-type",
                        choices=["webdataset", "csv", "auto"],
                        default="auto",
                        help="Which type of dataset to process.")

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help=
        "Number of samples in dataset. Required for webdataset if not available in info file.",
    )

    parser.add_argument("--k",
                        type=int,
                        default=None,
                        help="k for few shot ImageNet")
                        
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Default random seed.")

    parser.add_argument("--workers",
                        type=int,
                        default=8,
                        help="Number of dataloader workers per GPU.")

    parser.add_argument("--csv-separator",
                        type=str,
                        default="\t",
                        help="For csv-like datasets, which separator to use.")
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.")


    parser.add_argument(
        "--clip_load",
        type=str,
        default=None,
        help="Load finetuned clip",
    )

    parser.add_argument(
        "--wise_save",
        type=str,
        default=None,
        help="Save path for wiseft results",
    )

    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Repeated run number",
    )

    parser.add_argument("--get_labeled_csv",
                        default=False,
                        action="store_true",
                        help="get labels from csv.")
    
    parser.add_argument(
        "--supervised_label_key",
        type=str,
        default="label_idx",
        help="label key in csv.",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine scheduler",
    )
    #! lp-ft --------------------------
    parser.add_argument(
        "--head_path",
        type=str,
        default='',
        help="pre-trained head for lp-ft",
    )

   
    #! carot --------------------------
    parser.add_argument(
        "--distil_coef",
        type=float,
        default=0.0,
        help="coefficient for self-distillation loss",
    )

    parser.add_argument(
        "--ema_up_freq",
        type=int,
        default=0,
        help="required iterations for EMA teacher update",
    )

    parser.add_argument(
        "--m_sche_src",
        type=float,
        default=0.05,
        help="EMA teacher evolving schedule (src)",
    )

    parser.add_argument(
        "--m_sche_tar",
        type=float,
        default=0.9,
        help="EMA teacher evolving schedule (tar)",
    )

    parser.add_argument(
        "--distillation_temperature",
        type=float,
        default=1.0,
        help="Temperature for scaling teacher logits in distillation loss. "
             "Lower values make teacher predictions sharper, higher values make them softer. "
             "Recommended to set to half of teacher's effective temperature.",
    )

    parser.add_argument(
        "--m_warm_up",
        type=float,
        default=0.2,
        help="EMA teacher evolving schedule (warmup ratio)",
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Beta parameter for Beta distribution-based moving average (both alpha and beta of Beta distribution)",
    )
    
    parser.add_argument(
        "--ema_teacher",
        action="store_true",
        default=False,
        help=(
            "Use Exponential Moving Average (EMA) teacher instead of Beta Moving Average. "
            "Defaults: ema_up_freq=500, m_sche_tar=0.9, linear warmup from m_sche_src by 0.05 over first 20% iterations."
        ),
    )
    
    parser.add_argument(
        "--cross_fnorm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--l_orth_wv",
        type=float,
        default=0.0,
    )
    
    #! Knowledge Distillation Alpha Parameters --------------------------
    parser.add_argument(
        "--alpha_crd",
        type=float,
        default=0.0,
        help="Alpha coefficient for Contrastive Relational Distillation (CRD). "
             "Aligns contrastive distributions between teacher and student using KL divergence.",
    )
    
    parser.add_argument(
        "--alpha_fd",
        type=float,
        default=0.0,
        help="Alpha coefficient for Feature Distillation (FD). "
             "Direct alignment of visual and text embeddings using MSE loss.",
    )
    
    parser.add_argument(
        "--alpha_mfd",
        type=float,
        default=0.0,
        help="Alpha coefficient for Masked Feature Distillation (MFD). "
             "Similar to FD but uses masked images as input to student model.",
    )
    
    parser.add_argument(
        "--alpha_gd",
        type=float,
        default=0.0,
        help="Alpha coefficient for Gradient Distillation (GD). "
             "Aligns gradient information between teacher and student using analytical gradient computation.",
    )
    
    parser.add_argument(
        "--alpha_icl",
        type=float,
        default=0.0,
        help="Alpha coefficient for Interactive Contrastive Learning (ICL). "
             "Cross-modal contrastive learning where student embeddings contrast with teacher embeddings.",
    )
    
    parser.add_argument(
        "--alpha_afd",
        type=float,
        default=0.0,
        help="Alpha coefficient for Augmented Feature Distillation (AFD). "
             "Concatenates student and teacher embeddings, then applies linear fusion encoders.",
    )
    
    parser.add_argument(
        "--alpha_cross_kd",
        type=float,
        default=0.0,
        help="Alpha coefficient for Cross Knowledge Distillation (Cross KD). "
             "Uses cross-modal teacher-student interactions aligned with teacher's same-modal logits.",
    )
    
    parser.add_argument(
        "--alpha_temp_distil",
        type=float,
        default=0.0,
        help="Alpha coefficient for Temperature-scaled Distillation. "
             "Uses temperature scaling for teacher logits in distillation loss (current implementation).",
    )
    
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        help="Masking ratio for Masked Feature Distillation (MFD). "
             "Fraction of image patches to mask during training.",
    )
    #! ---------------
    parser.add_argument(
        "--wb_project",
        type=str,
        default="",
        help="weight and bias project name",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="",
        help="zs / ft / flyp / lpft / carot ...",
    )

    parser.add_argument(
        "--use_fp16",
        type=int,
        default=1,
        help="mixed precision training flag",
    )

    parser.add_argument(
        "--temperature_scale",
        type=float,
        default=0.0,
        help="temperature scaling",
    )
    parser.add_argument(
        "--full_eval",
        type=int,
        default=0,
        help="temperature scaling",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )

    parser.add_argument(
        "--supcon_temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for SupCon loss when labels are available",
    )

    # Gradient clipping parameters
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0001,
        help="Maximum gradient norm for gradient clipping (0.0 to disable)",
    )

    parser.add_argument(
        "--grad_norm_multiplier",
        type=float,
        default=100.0,
        help="Multiplier for maximum gradient norm for gradient clipping",
    )
    
    # Layer freezing parameters
    parser.add_argument(
        "--freeze_text_encoder",
        action="store_true",
        default=False,
        help="Freeze the text encoder (transformer) parameters. Default: False (trainable)",
    )
    
    parser.add_argument(
        "--trainable_layers",
        type=int,
        default=-1,
        help="Number of last layers to keep trainable in both encoders. -1 means all layers trainable, 0 means all frozen, N means last N layers trainable",
    )

    parser.add_argument(
        "--ew_sd_alpha",
        type=float,
        default=0.0,
        help="Eigenvalue weighting parameter for EW-SD. 0=uniform (standard SD), 1=covariance-normalized, 2=maximum adaptation. Default: 0.0 (disabled)",
    )

    parser.add_argument(
        "--ew_sd_lambda",
        type=float,
        default=0.0,
        help="Overall regularization strength for EW-SD. Default: 0.0 (disabled)",
    )

    parser.add_argument("--use-spatial-captions", type=int, default=0,
                        help="Master switch for spatial-caption geodesic mixing (0 = inert).")
    parser.add_argument("--spatial-captions-jsonl", type=str, default=None,
                        help="JSONL with keys 'image' and 'dense_caption'.")
    parser.add_argument("--alpha-tt-mix", type=float, default=0.2,
                        help="Beta(alpha, .) for Scenario-alpha text-text mix; U-shape when < 1.")
    parser.add_argument("--beta-tt-mix", type=float, default=0.2,
                        help="Beta(., beta) for Scenario-alpha text-text mix.")
    parser.add_argument("--tt-per-sample", type=int, default=0,
                        help="1 = one lambda per sample; 0 = one per batch.")
    parser.add_argument("--beta-mix-coef", type=float, default=0.0,
                        help="Weight for Scenario-beta hard negatives; 0 = alpha-only.")
    parser.add_argument("--alpha-it-mix", type=float, default=0.5,
                        help="Symmetric Beta(alpha, alpha) for Scenario-beta image-text mix.")
    parser.add_argument("--it-per-sample", type=int, default=0,
                        help="1 = one lambda per sample; 0 = one per batch.")
    parser.add_argument("--beta-mix-target", type=str, default="spatial",
                        choices=["spatial", "template", "alpha_mixed"],
                        help="Text vector to bridge with the image in Scenario beta.")
    parser.add_argument("--tau2", type=float, default=0.0,
                        help="Literal m_tau for Scenario-beta off-diagonal negatives "
                             "(reference default 0.01); <=0 reuses the learned logit scale.")
    parser.add_argument("--sanity-check", action="store_true", default=False,
                        help="Assert geodesic-mix invariants on the first batch.")

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args


# # 1. Freeze entire text encoder (ALL text components)
# python train.py --method carot --freeze_text_encoder

# # 2. Fine-tune only last 2 transformer layers + embeddings/projections  
# python train.py --method carot --trainable_layers 2

# # 3. Extreme fine-tuning: only logit_scale trainable
# python train.py --method carot --trainable_layers 0

# # 4. Text frozen + only vision embeddings/projections trainable
# python train.py --method carot --freeze_text_encoder --trainable_layers 0