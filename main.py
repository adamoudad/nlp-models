from pathlib import Path

from models import prediction_lstm
from feed import NextTokenGenerator

data_generator = NextTokenGenerator(
    data_file=Path('sample_data.txt'),
    batch_size=4
)

# model = prediction_lstm(10, 6)
