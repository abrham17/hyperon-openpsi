from hyperon import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

metta = MeTTa()

metta.run("""
!(register-module! ../../../hyperon-openpsi)
!(register-module! ../../utilities-module)
!(import! &self utilities-module:utils)
!(import! &self hyperon-openpsi:main:emotion:emotion)
!(bind! &emotion-space (new-space))
""")



def fetch_emotions():
    emotions = metta.run('!(get-emotions &emotion-space)')[0]
    emotion_list = []
    emotion_list_value = []
    for e in emotions[0].get_children():
        emotion_list.append(str(e.get_children()[1]))
        emotion_list_value.append(float(str(e.get_children()[2])))
    return emotion_list, emotion_list_value

fig, ax = plt.subplots()

def animate(frame):

    labels, values = fetch_emotions()

    ax.clear()
    ax.bar(labels, values, color='skyblue')
    ax.set_ylim(0, 1) 
    ax.set_title('Real-Time Emotion Visualization')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Values')
    plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=1000)

plt.show()
