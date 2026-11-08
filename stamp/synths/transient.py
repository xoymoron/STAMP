from stamp.models.tcn import TCN


class TransientTCN(TCN):
    """Waveform TCN without a learned excitation or temporal conditioning."""
