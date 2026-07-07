cfg = dict(
    _BASE_ = ['../ADF_dinov2.py'],
    exp_name='UCOD-ADF-confident',
    train_cfg=dict(
        max_epoch=30,
        lr0=1e-4,
        step_lr_size=15,
        step_lr_gamma=0.8,
        save_cfg=dict(
            save_interval=2,
            start_save=5,
        ),
        adf_cfg=dict(
            lambda_conf=0.70,
            lambda_region=0.05,
            lambda_temporal=0.05,
            qaa_delta=0.20,
            memory_alpha=0.85,
            memory_warmup=2,
        ),
    ),
    val_cfg=dict(
        val_interval=2,
        val_start=2,
    ),
)
