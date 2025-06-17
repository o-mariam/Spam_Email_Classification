from .spam_client import SpamDetectorClient
from .neural_client import SpamDetectorClientNeural
from .transformer_e5_client import SpamDetectorClientE5
from .transformer_emotion_client import SpamDetectorClientEmotion

neural_client = SpamDetectorClientNeural()
e5_client = SpamDetectorClientE5()
emotion_client = SpamDetectorClientEmotion()