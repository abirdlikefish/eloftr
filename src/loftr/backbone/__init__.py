from .backbone import RepVGG_8_1_align

def build_backbone(config):
    # v7_pcclahe: read cfg.LOFTR.BACKBONE_IN_CHANNELS (default 1, byte-identical
    # to v0-v6.1; v7 cfg overrides to 2). config here is lower_config(LOFTR), so
    # we use .get() to keep working even on older configs that never declared
    # the field (it falls back to the default.py literal default of 1).
    in_channels = config.get('backbone_in_channels', 1)
    if config['backbone_type'] == 'RepVGG':
        if config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return RepVGG_8_1_align(config['backbone'], in_channels=in_channels)
        else:
            raise ValueError(f"LOFTR.ALIGN_CORNER {config['align_corner']} not supported.")
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
