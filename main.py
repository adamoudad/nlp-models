from pathlib import Path

import numpy as np
from lstm import prediction_lstm

# data_generator = NextTokenGenerator(
#     data_file=Path('sample_data.txt'),
#     batch_size=4
# )

from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer('./bert-base-uncased-vocab.txt', lowercase=True)
sample_data = Path('./data_sample.txt').read_text()
output = tokenizer.encode(sample_data)

n_classes = max(output.ids) + 1

model = prediction_lstm(n_classes)

training_samples = np.array([ output.ids[i:i+10] for i in range(len(output.ids) - 11) ])

training_labels = np.zeros((len(output.ids) - 11, n_classes))
for i in range(11, len(output.ids)):
    training_labels[i-11,output.ids[i]] = 1.

model.fit(x=training_samples, y=training_labels, epochs=50)

for sample in training_samples:
    token_prediction = tokenizer.id_to_token(model.predict(np.expand_dims(sample, axis=0))[0].argmax())
    print("Input:", tokenizer.decode(sample))
    print("Pred:", token_prediction)
