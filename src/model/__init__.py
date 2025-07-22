from src.model.ours import EncoderOnlySegmenterPL
from src.model.naive import NaiveSegmenterPL
from src.model.ifa import IFASegmenterPL
from src.model.ssp import SSPSegmenterPL

REGISTERED_MODELS = {
    'ours': EncoderOnlySegmenterPL,
    'ifa': IFASegmenterPL,
    'naive': NaiveSegmenterPL,
    'ssp': SSPSegmenterPL,
}