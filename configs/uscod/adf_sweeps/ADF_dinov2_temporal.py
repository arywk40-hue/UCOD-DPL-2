cfg = dict(
    _BASE_ = ['../ADF_dinov2.py'],
    exp_name='UCOD-ADF-temporal',
    train_cfg=dict(
        max_epoch=40,
        lr0=5e-5,
        step_lr_size=20,
        step_lr_gamma=0.8,
        save_cfg=dict(
            save_interval=2,
            start_save=5,
        ),
        adf_cfg=dict(
            lambda_conf=0.25,
            lambda_region=0.05,
            lambda_temporal=0.50,
            qaa_delta=0.30,
            memory_alpha=0.98,
            memory_warmup=5,
        ),
    ),
    val_cfg=dict(
        val_interval=2,
        val_start=2,
    ),
)
