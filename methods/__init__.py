
from .fedavg import *
from .local import *
from .fedpac import *

def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'Local':LocalUpdate_StandAlone,
                   'FedPAC':LocalUpdate_FedPAC,
    }

    return LocalUpdate[rule]