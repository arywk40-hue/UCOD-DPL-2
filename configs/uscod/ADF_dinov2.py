cfg = dict(
    _BASE_ = [
        '../__base__/accelerate.py',
        '../__base__/newbase.py',
        '../dataset/cod4040.py'
    ],
    exp_name = 'UCOD-ADF',
    train_cfg = dict(
        max_epoch=25,
        start_epoch=0,
        lr0=2e-4,
        step_lr_size=25,
        step_lr_gamma=0.95,
        # ADF replaces the discriminator — no dis_* configs needed
        merge_method='adf',
        # Agentic Decision Framework configuration
        adf_cfg = dict(
            # ── Loss weights ──
            lambda_conf=0.5,       # confidence calibration (QAA)
            lambda_region=0.1,     # region consistency (RAA)
            lambda_temporal=0.3,   # temporal consistency (Memory)
            
            # ── Agent 1: QAA ──
            qaa_delta=0.3,         # agreement threshold for self-supervised target
            qaa_hidden_dim=64,     # hidden channels in conv layers
            
            # ── Agent 2: RAA ──
            raa_tau_variance=0.1,  # feature variance threshold for "easy"
            raa_tau_boundary=0.5,  # boundary gradient threshold for "easy"
            raa_tau_size=0.02,     # size ratio threshold for "ambiguous"
            raa_tau_ambiguous=0.3, # feature variance threshold for "ambiguous"
            
            # ── Agent 3: Memory ──
            memory_alpha=0.95,     # EMA decay (slow update for trends)
            memory_warmup=3,       # minimum visits before stability is meaningful
            
            # ── Agent 4: Decision Agent ──
            da_hidden_dim=64,      # hidden channels
            da_mid_dim=32,         # intermediate channels before heads
            da_lt_threshold=0.5,   # Look-Twice trigger threshold
        ),
    ),
    val_cfg = dict(
        look_twice=True,
        look_twice_th=0.15,
        expand_type='dynamic',
        val_interval=5,
        val_start=5,
    ),
    log_cfg = dict(
        log_interval=50,
    ),
    model_cfg=dict(
        ema_weight=0.99,
        dim=768,
        feature_size=68,
        # dis_use_features is not needed for ADF — kept for backward compat
        dis_use_features=False,
    ),
    dataset_cfg=dict(
        cache_dir='./datasets/cache',
        val_loader_cfg = dict(
            batch_size=1,
            num_workers=0,
            shuffle=False
        ),
        trainloader_cfg = dict(
            batch_size=16,
            num_workers=0,
            shuffle=True
        ),
        valset_cfg = dict(
            DATASET='TE-CAMO',
            require_label=True,
            image_size=(518, 518),
        ),
        trainset_cfg = dict(
            DATASET='TR-CAMO+TR-COD10K',
            image_size=(518, 518),
            require_label=False,
            bkg_th=0.6,
        ),
        feature_extractor_cfg=dict(
            type='dinov2',
            backbone_weight_base='~/workspace/weights/huggingface',
            backbone='facebook/dinov2-base',
            backbone_weights='./weights',
            backbone_type='huggingface',
            backbone_feat_dim=[768],
        ),
    )
)
