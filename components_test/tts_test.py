from gtts import gTTS
import pygame

tts = gTTS(text='May aso sa daan!', lang='tl')
tts.save("welcome.mp3")

pygame.mixer.init()
pygame.mixer.music.load("welcome.mp3")
pygame.mixer.music.play()
