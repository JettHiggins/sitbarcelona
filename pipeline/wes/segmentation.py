"""Single Wes segmentation path: constrained dead-ball revision over Mall evidence.

The selected algorithm is the former joint-d experiment. Mall's H/A decisions
remain fixed wherever they are already live; only uncertain D windows can be
reopened. No competition labels are read here.
"""

from .joint import predict_joint


def predict(z, F, L2, L3, L4):
    return predict_joint(z, F, L2, L3, L4)
