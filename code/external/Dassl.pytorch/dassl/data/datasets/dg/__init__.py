from .pacs import PACS
from .vlcs import VLCS
try:
    from .wilds import *
except ModuleNotFoundError:
    pass
from .cifar_c import CIFAR10C, CIFAR100C
from .digits_dg import DigitsDG
from .digit_single import DigitSingle
from .office_home_dg import OfficeHomeDG
