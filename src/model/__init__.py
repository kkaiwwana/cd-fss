from src.model.ours import EncoderOnlySegmenterPL
from src.model.naive import NaiveSegmenterPL
from src.model.ifa import IFASegmenterPL

REGISTERED_MODELS = {
    'ours': EncoderOnlySegmenterPL,
    'ifa': IFASegmenterPL,
    'naive': NaiveSegmenterPL,
}