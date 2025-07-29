import joblib
from config import DEVELOPMENT_STAGES

stage_vocab = {stage: idx for idx, stage in enumerate(DEVELOPMENT_STAGES)}
joblib.dump(stage_vocab, 'models/stage_vocab.pkl')