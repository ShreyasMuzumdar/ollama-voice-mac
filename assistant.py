import sys
import json
import wave
import time
import pyttsx3
import torch
import requests
import soundfile
import yaml
import pygame
import pygame.locals
import numpy as np
import pyaudio
import whisper
import logging
import threading
import queue
import onnxruntime as ort  # Import ONNX Runtime for wake word detection

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants for the application
BACK_COLOR = (0, 0, 0)  # Background color (black)
REC_COLOR = (255, 0, 0)  # Recording indicator color (red)
TEXT_COLOR = (255, 255, 255)  # Text color (white)
REC_SIZE = 80  # Size of the recording indicator circle
FONT_SIZE = 24  # Font size for displayed text
WIDTH = 320  # Width of the display window
HEIGHT = 240  # Height of the display window
KWIDTH = 20  # Width of sound energy bars
KHEIGHT = 6  # Height of sound energy bars
MAX_TEXT_LEN_DISPLAY = 32  # Maximum length of text to display on screen

# Audio input configuration
INPUT_DEFAULT_DURATION_SECONDS = 5  # Default duration for audio input
INPUT_FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
INPUT_CHANNELS = 1  # Number of audio channels (mono)
INPUT_RATE = 8000  # Sampling rate (16 kHz)
INPUT_CHUNK = 2048  # Number of frames per buffer
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'}  # Headers for OLLaMa API requests
INPUT_CONFIG_PATH = "assistant.yaml"  # Path to the configuration file

# Define the Assistant class
class Assistant:
    def __init__(self):
        # Initialize the Assistant
        logging.info("Initializing Assistant")
        self.config = self.init_config()  # Load configuration from YAML file

        # Load the program icon
        programIcon = pygame.image.load('assistant.png')

        # Initialize Pygame display and clock
        self.clock = pygame.time.Clock()
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("Assistant")

        # Create the display window
        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.font = pygame.font.SysFont(None, FONT_SIZE)  # Set the font for text rendering

        # Initialize PyAudio for audio input
        self.audio = pyaudio.PyAudio()

        # Initialize the text-to-speech engine
        self.tts = pyttsx3.init("nsss")
        self.tts.setProperty('rate', self.tts.getProperty('rate') - 20)  # Adjust speech rate

        # Load the ONNX model for wake word detection
        self.wake_word_model_path = "/Users/shreyas/ollama-voice-mac/hey_jarvis.onnx"  # Path to your ONNX file
        self.session = ort.InferenceSession(self.wake_word_model_path)  # Load the ONNX model
        self.input_name = self.session.get_inputs()[0].name  # Get input name
        self.output_name = self.session.get_outputs()[0].name  # Get output name

        # Test audio input availability
        try:
            self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except Exception as e:
            logging.error(f"Error opening audio stream: {str(e)}")
            self.wait_exit()  # Exit if no audio input is available

        # Load the Whisper model for speech recognition
        self.display_message(self.config.messages.loadingModel)
        self.model = whisper.load_model(self.config.whisperRecognition.modelPath)
        self.context = []  # Initialize conversation context

        # Greet the user and prompt for input
        self.text_to_speech(self.config.conversation.greeting)
        time.sleep(0.5)
        self.display_message(self.config.messages.pressSpace)

    def wait_exit(self):
        # Display an error message and wait for the user to exit
        while True:
            self.display_message(self.config.messages.noAudioInput)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        # Shutdown the Assistant and release resources
        logging.info("Shutting down Assistant")
        self.audio.terminate()  # Close PyAudio
        pygame.quit()  # Close Pygame
        sys.exit()  # Exit the program

    def init_config(self):
        # Load configuration from the YAML file
        logging.info("Initializing configuration")
        class Inst:
            pass  # Dummy class to hold configuration values

        with open('assistant.yaml', encoding='utf-8') as data:
            configYaml = yaml.safe_load(data)  # Load YAML data

        config = Inst()
        config.messages = Inst()
        config.messages.loadingModel = configYaml["messages"]["loadingModel"]  # Loading model message
        config.messages.pressSpace = configYaml["messages"]["pressSpace"]  # Press space message
        config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]  # No audio input message
        config.messages.listeningForWakeWord = configYaml["messages"]["listeningForWakeWord"]  # Wake word message

        config.conversation = Inst()
        config.conversation.greeting = configYaml["conversation"]["greeting"]  # Greeting message

        config.ollama = Inst()
        config.ollama.url = configYaml["ollama"]["url"]  # OLLaMa API URL
        config.ollama.model = configYaml["ollama"]["model"]  # OLLaMa model name

        config.whisperRecognition = Inst()
        config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]  # Whisper model path
        config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]  # Language for speech recognition

        return config

    def display_rec_start(self):
        # Display a red circle to indicate recording
        logging.info("Displaying recording start")
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH / 2, HEIGHT / 2), REC_SIZE)
        pygame.display.flip()

    def display_sound_energy(self, energy):
        # Display sound energy as vertical bars
        logging.info(f"Displaying sound energy: {energy}")
        COL_COUNT = 5
        RED_CENTER = 100
        FACTOR = 10
        MAX_AMPLITUDE = 100

        self.windowSurface.fill(BACK_COLOR)
        amplitude = int(MAX_AMPLITUDE * energy)
        hspace, vspace = 2 * KWIDTH, int(KHEIGHT / 2)

        def rect_coords(x, y):
            return (int(x - KWIDTH / 2), int(y - KHEIGHT / 2), KWIDTH, KHEIGHT)

        for i in range(-int(np.floor(COL_COUNT / 2)), int(np.ceil(COL_COUNT / 2))):
            x, y, count = WIDTH / 2 + (i * hspace), HEIGHT / 2, amplitude - 2 * abs(i)

            mid = int(np.ceil(count / 2))
            for i in range(0, mid):
                offset = i * (KHEIGHT + vspace)
                pygame.draw.rect(self.windowSurface, RED_CENTER, rect_coords(x, y + offset))
                # Mirror the bars
                pygame.draw.rect(self.windowSurface, RED_CENTER, rect_coords(x, y - offset))
        pygame.display.flip()

    def display_message(self, text):
        # Display a text message on the screen
        logging.info(f"Displaying message: {text}")
        self.windowSurface.fill(BACK_COLOR)

        # Render the text and center it on the screen
        label = self.font.render(text
                                 if (len(text) < MAX_TEXT_LEN_DISPLAY)
                                 else (text[0:MAX_TEXT_LEN_DISPLAY] + "..."),
                                 1,
                                 TEXT_COLOR)

        size = label.get_rect()[2:4]
        self.windowSurface.blit(label, (WIDTH / 2 - size[0] / 2, HEIGHT / 2 - size[1] / 2))

        pygame.display.flip()

    def waveform_from_mic(self, key=pygame.K_SPACE) -> np.ndarray:
        # Capture audio from the microphone while the space key is pressed
        logging.info("Capturing waveform from microphone")
        self.display_rec_start()

        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        while True:
            pygame.event.pump()  # Process event queue
            pressed = pygame.key.get_pressed()
            if pressed[key]:
                data = stream.read(INPUT_CHUNK)
                frames.append(data)
            else:
                break

        stream.stop_stream()
        stream.close()

        # Convert the audio data to a numpy array
        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        # Convert speech to text using the Whisper model
        logging.info("Converting speech to text")
        result_queue = queue.Queue()

        def transcribe_speech():
            try:
                logging.info("Starting transcription")
                transcript = self.model.transcribe(waveform,
                                                language=self.config.whisperRecognition.lang,
                                                fp16=torch.cuda.is_available())
                logging.info("Transcription completed")
                text = transcript["text"]
                print('\nMe:\n', text.strip())
                result_queue.put(text)
            except Exception as e:
                logging.error(f"An error occurred during transcription: {str(e)}")
                result_queue.put("")

        transcription_thread = threading.Thread(target=transcribe_speech)
        transcription_thread.start()
        transcription_thread.join()

        return result_queue.get()

    def preprocess_audio(self, audio_data):
        """
        Preprocess the audio data to match the input requirements of the ONNX model.
        """
        # Normalize the audio data to the range [-1, 1]
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Reshape the audio data to match the input shape of the ONNX model
        # Example: If the model expects [batch_size, 16, 96], we need to transform the data
        # Here, we assume the model expects [1, 16, 96]
        target_channels = 16
        target_sequence_length = 96

        # Pad or truncate the audio data to the target sequence length
        if len(audio_data) < target_sequence_length:
            # Pad with zeros if the audio is too short
            audio_data = np.pad(audio_data, (0, target_sequence_length - len(audio_data)), mode='constant')
        else:
            # Truncate if the audio is too long
            audio_data = audio_data[:target_sequence_length]

        # Reshape the audio data to match the input shape [1, 16, 96]
        # Here, we repeat the audio data across the channel dimension
        processed_audio = np.tile(audio_data, (target_channels, 1))  # Shape: [16, 96]
        processed_audio = np.expand_dims(processed_audio, axis=0)  # Add batch dimension: [1, 16, 96]

        # Print the shape for debugging
        logging.info(f"Processed audio shape: {processed_audio.shape}")

        return processed_audio

    def detect_wake_word(self, audio_data):
        """
        Detect the wake word using the ONNX model.
        """
        try:
            # Preprocess the audio data
            processed_audio = self.preprocess_audio(audio_data)

            # Run inference using the ONNX model
            result = self.session.run([self.output_name], {self.input_name: processed_audio})

            # Check if the wake word was detected
            if result[0] > 0.5:  # Adjust the threshold as needed
                logging.info("Wake word detected!")
                return True
            return False
        except Exception as e:
            logging.error(f"Error in wake word detection: {str(e)}")
            return False

    def listen_for_wake_word(self):
        # Continuously listen for the wake word
        logging.info("Listening for wake word")
        self.display_message(self.config.messages.listeningForWakeWord)

        stream = self.audio.open(format=INPUT_FORMAT,
                                channels=INPUT_CHANNELS,
                                rate=INPUT_RATE,
                                input=True,
                                frames_per_buffer=INPUT_CHUNK)

        while True:
            data = stream.read(INPUT_CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) * (1 / 32768.0)

            if self.detect_wake_word(audio_data):
                logging.info("Wake word detected")
                stream.stop_stream()
                stream.close()
                return True

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    logging.info("Quit key pressed during wake word detection")
                    stream.stop_stream()
                    stream.close()
                    self.shutdown()
                    return False

    def ask_ollama(self, prompt, responseCallback):
        # Send a prompt to the OLLaMa API and handle the response
        logging.info(f"Asking OLLaMa with prompt: {prompt}")
        full_prompt = prompt if hasattr(self, "contextSent") else (prompt)
        self.contextSent = True
        jsonParam = {
            "model": self.config.ollama.model,
            "stream": True,
            "context": self.context,
            "prompt": full_prompt
        }

        try:
            response = requests.post(self.config.ollama.url,
                                    json=jsonParam,
                                    headers=OLLAMA_REST_HEADERS,
                                    stream=True,
                                    timeout=30)  # Increase the timeout value
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                body = json.loads(line)
                token = body.get('response', '')
                full_response += token

                if 'error' in body:
                    logging.error(f"Error from OLLaMa: {body['error']}")
                    responseCallback("Error: " + body['error'])
                    return

                if body.get('done', False) and 'context' in body:
                    self.context = body['context']
                    break

            responseCallback(full_response.strip())

        except requests.exceptions.ReadTimeout as e:
            logging.error(f"ReadTimeout occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, the request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, an error occurred. Please try again.")

    def text_to_speech(self, text):
        # Convert text to speech using the TTS engine
        logging.info(f"Converting text to speech: {text}")
        print('\nAI:\n', text.strip())

        def play_speech():
            try:
                logging.info("Initializing TTS engine")
                engine = pyttsx3.init()

                # Adjust the speech rate (optional)
                rate = engine.getProperty('rate')
                engine.setProperty('rate', rate - 50)  # Decrease the rate by 50 units

                # Add a short delay before converting text to speech
                time.sleep(0.5)  # Adjust the delay as needed

                logging.info("Converting text to speech")
                engine.say(text)
                engine.runAndWait()
                logging.info("Speech playback completed")
            except Exception as e:
                logging.error(f"An error occurred during speech playback: {str(e)}")

        speech_thread = threading.Thread(target=play_speech)
        speech_thread.start()

def main():
    # Main function to run the Assistant
    logging.info("Starting Assistant")
    pygame.init()

    ass = Assistant()

    push_to_talk_key = pygame.K_SPACE
    quit_key = pygame.K_ESCAPE

    while True:
        ass.clock.tick(60)

        # Listen for wake word
        if ass.listen_for_wake_word():
            ass.display_message(ass.config.messages.pressSpace)

            while True:
                ass.clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == push_to_talk_key:
                            logging.info("Push-to-talk key pressed")
                            speech = ass.waveform_from_mic(push_to_talk_key)
                            transcription = ass.speech_to_text(waveform=speech)
                            ass.ask_ollama(transcription, ass.text_to_speech)
                            time.sleep(1)
                            ass.display_message(ass.config.messages.pressSpace)

                        elif event.key == quit_key:
                            logging.info("Quit key pressed")
                            ass.shutdown()

if __name__ == "__main__":
    main()