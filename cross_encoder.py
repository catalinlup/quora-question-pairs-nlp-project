from sentence_transformers import CrossEncoder
from dataset import question1_validation, question2_validation, validation_data
from utils import save_feature, load_feature
import numpy as np
model = CrossEncoder("cross-encoder/qnli-distilroberta-base")

question_pairs = list(zip(question1_validation, question2_validation))

scores = model.predict(question_pairs)

save_feature(scores, 'qnli_cross_encoder_scores')

# scores = load_feature('qnli_cross_encoder_scores')
print(scores)

final_predictions = (scores > 0.93).astype(np.int32)

ids = validation_data['id'].to_numpy().astype(dtype=np.int64)
predictions = final_predictions.reshape((final_predictions.shape[0], 1))
ids = ids.reshape((ids.shape[0], 1))

result = np.concatenate([ids, predictions], axis=1)


np.savetxt('predictions/qnli_cross_encoder.csv', result, delimiter=',', fmt='%d')