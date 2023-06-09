from gtts import gTTS
import pygame

tts = gTTS(text='This is a test', lang='tl')
tts.save("test.mp3")

pygame.mixer.init()
pygame.mixer.music.load("welcome.mp3")
pygame.mixer.music.play()