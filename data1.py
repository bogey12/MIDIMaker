import os
import pygame
import midi
import random
import numpy as np
import json
import tensorflow as tf
import rater
#import pyglet

np.set_printoptions(threshold='nan')
#files = ['music/' + f for f in os.listdir('music')]
ratings = {}

lowerBound = 44
upperBound = 80

CACHE = json.load(open('test.json', 'r'))

def midi_to_matrix(midi_file, data):

    '''opens a midi file and then, for every tick in the file, output an array 
    containing all the note on events '''

    pathname = os.path.join(data, midi_file)

    pattern = midi.read_midifile(pathname)

    time_left = [track[0].tick for track in pattern]

    poses = [0 for _ in pattern]

    state_matrix = []
    span = upperBound - lowerBound
    time = 0

    state = [[0, 0] for _ in range(span)]
    state_matrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            old_state = state
            state = [[old_state[x][0], 0] for x in range(span)]
            state_matrix.append(state)

        for i in range(len(time_left)):
            while time_left[i] == 0:
                track = pattern[i]
                pos = poses[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return state_matrix

                try:
                    time_left[i] = track[pos + 1].tick
                    poses[i] += 1
                except IndexError:
                    time_left[i] = None

            if time_left[i] is not None:
                time_left[i] -= 1

        if all(t is None for t in time_left):
            break

        time += 1

    return state_matrix
def noteStateMatrixToMidi(statematrix, name="example"):

    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
        print offNotes
        print 'offNotes'
        print onNotes
        print 'onNotes'
        prevstate = state
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

def next_batch(data, number):
    """
    Gets a human to rate 10 songs, then returns training data in the following format
    [
        (rating0, song0 in 128x256 flattened matrix),
        (rating1, song1 in 128x256 flattened matrix),
        ...
    ]
    :return:
    """
    batch = []
    
    for x in random.sample(os.listdir(data), number): 
        if x in CACHE:
            rating = CACHE[x]
        else:
            try:
                pass
                play_midi(x, data)
                
            except Exception as e:
                print(e)
                continue
            rating = int(input('Rate this song: '))
            CACHE[x] = rating

        try:
            a = np.array(midi_to_matrix(x, data))
            #print a.shape
            #print a
        except Exception as e:
            print(e)
            continue

        b = a.flatten()[:112*36]
        if len(b) < 112*36: continue
            #b = np.resize(b, (56*76))
        tup1 = (np.array([rating]), b)
        batch.append(tup1)

    batch = (
        map(lambda x: x[0], batch),
        map(lambda x: x[1], batch)
    )
    with open('test.json', 'w') as test:
        json.dump(CACHE, test)

    return batch

def next_batch_cut(data, number):

    batch = []
    
    for x in random.sample(os.listdir(data), number): 
        if x in CACHE:
            rating = CACHE[x]
        else:
            try:
                pass
                play_midi(x, data)
                
            except Exception as e:
                print(e)
                continue
            rating = int(input('Rate this song: '))
            CACHE[x] = rating

        try:
            a = np.array(midi_to_matrix(x, data))
            #print a.shape
            #print a
        except Exception as e:
            print(e)
            continue

        b = a.flatten()[:112*76]
        if len(b) < 112*76: continue

        '''for f in range(56):
            s = f * 76 * 2
            t = s + 76 * 2
            d = b[s:t]'''

        
            #b = np.resize(b, (56*76))
        tup1 = (np.array([rating]), b)
        batch.append(tup1)

    batch = (
        map(lambda x: x[0], batch),
        map(lambda x: x[1], batch)
    )
    with open('test.json', 'w') as test:
        json.dump(CACHE, test)

    return batch



def generate():
    with tf.Session() as sess:
        i=0
        a = rater.Rater()
        #a.saver.restore(sess, 'saved_models/rater.ckpt')
        sess.run(tf.global_variables_initializer())
        print a.W_fc2.eval()
        for g in range(100):
            batch = next_batch('music', 1)
            matrix = a.y_conv2.eval(feed_dict={a.x2: batch[1], a.keep_prob: 0.5})
            '''for x in np.nditer(matrix, flags=['buffered'], op_flags = ['readwrite']):
                    prob = x * 100
                    roll = np.random.randint(100)
                    chance = np.random.randint(1000)
                    if prob < roll:
                        x[...] = 0
                    else:
                        if  prob < chance:
                            x[...] = 0
                        else:
                            x[...] = 1'''

            matrix = np.resize(matrix, (200*76*2))
            
            '''for c in range(1000):
                matrix1 = a.y_conv2.eval(feed_dict={a.x2: batch[1], a.keep_prob: 0.5})[2*78*28:]
                matrix = np.append(matrix, matrix1)
                del matrix1'''
            try:
                matrix = np.reshape(matrix, (-1, 76, 2))
                #print matrix
                #noteStateMatrixToMidi(matrix, 'output/testrestssa'+str(i))
            except Exception as e:
                print(e)
                continue
            i += 1


def play_midi(filename, direct):
    pathname = os.path.join(direct, filename)
    explosion = pyglet.media.load(pathname, streaming=False)
    #explosion.play()
    #pygame.init()
    #pygame.mixer.music.load(pathname)
    #pygame.mixer.music.play()
    #pygame.quit()
    pass


if __name__ == '__main__':
    for file in files:
        play_midi(file)
        rating = int(input('Rate this song: '))
        ratings[file] = rating
