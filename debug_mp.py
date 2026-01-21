import mediapipe as mp
try:
    print(f"MediaPipe Version: {mp.__version__}")
    mp_hands = mp.solutions.hands
    print("mp.solutions.hands imported successfully")
except AttributeError as e:
    print(f"Error accessing mp.solutions.hands: {e}")
except Exception as e:
    print(f"General Error: {e}")
