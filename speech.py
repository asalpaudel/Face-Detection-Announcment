
import time
import random
import threading
from gtts import gTTS
import os
import tempfile
from datetime import datetime
import pygame 
# from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI
welcome_templates = [
    "विरिन्ची कलेजमा {name} लाई स्वागत छ।",
    "विरिन्ची कलेजमा {name} लाई हार्दिक स्वागत छ।",
    "विरिन्ची कलेजमा {name} लाई पुनः स्वागत छ।",
    "विरिन्ची कलेजमा {name} लाई स्वागत गर्न पाउँदा खुशी लाग्यो।",
    "विरिन्ची कलेजमा {name} लाई हामी स्वागत गर्दछौं।",
    "विरिन्ची कलेजमा {name} लाई देखेर खुशी लाग्यो।",
    "विरिन्ची कलेजमा {name} को आगमनमा स्वागत छ।",
    "विरिन्ची कलेजमा {name} लाई भेट्दा हर्षित छौं।",
    "विरिन्ची कलेजको तर्फबाट {name} लाई स्वागत छ।",
    "विरिन्ची कलेजमा उपस्थित {name} लाई हार्दिक स्वागत छ।"
]
# Speak the welcome message in Nepali
def speak_nepali(text):
    try:
        language = 'ne'
        speech = gTTS(text=text, lang=language, slow=False)
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        speech.save(temp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # Stop and unload music
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        os.remove(temp_path)
    except Exception as e:
        print(f"❌ Error during TTS: {e}")



def welcome_person(name):
    message = random.choice(welcome_templates).format(name=name)
    print(f"[DEBUG] Speaking: {message}")
    threading.Thread(target=speak_nepali, args=(message,)).start()

# Function to welcome recognized person
# def welcome_person(name):
#     message = f"विरिन्ची कलेजमा {name} लाई स्वागत छ।"
#     threading.Thread(target=speak_nepali, args=(message,)).start()
