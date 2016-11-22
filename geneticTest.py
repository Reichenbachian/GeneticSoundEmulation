import wave
import pyaudio
from Sound import Sound
import random
import numpy as np
from deap import creator, base, tools, algorithms
from threading import Thread
import matplotlib.pyplot as plt
import sys
import struct

#GLOBALS
p = pyaudio.PyAudio()
volume = 1
samples=[]
numWaves=25
daChar = [' ']
population=None
generation = 0
record = True
#LOAD AUDIO NP
wf = wave.open('PianoNoises/39198__jobro__piano-ff-050.wav', 'rb')
signal = wf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
signal = signal/signal.max()
#1: Amplitude 2: Pitch 3: decay factor 4: type(0-.3sin, .3-.6cos, .6-1saw) 5:Shift 6: height
numFeatures = 7
featureCoefs = {"Amplitude": 1/numWaves, "Pitch": 50000,
			"Decay": 15000,"Type": 1,"Shift": 200, "Height": 1/numWaves, "Slope": 1}
def checkInput():
	while True:
		daChar[0] = input()
		if daChar[0] == 'q':
			with open("log.txt", "a") as f:
				f.write("-----------------------------------------\n")
			sys.exit()

def getWave(arr):
	sample = Sound(arr[0]*featureCoefs['Amplitude'],
				   arr[1]*featureCoefs['Pitch'],
				   arr[2]*featureCoefs['Decay'],
				   arr[3]*featureCoefs['Type'],
				   arr[4]*featureCoefs['Shift'],
				   arr[5]*featureCoefs['Height'],
				   arr[6]*featureCoefs['Slope']).getWave()
	for i in range(numFeatures, len(arr), numFeatures):
		sample += Sound(arr[i]*featureCoefs['Amplitude'],
					  arr[i+1]*featureCoefs['Pitch'],
					  arr[i+2]*featureCoefs['Decay'],
					  arr[i+3]*featureCoefs['Type'],
					  arr[i+4]*featureCoefs['Shift'],
					  arr[i+5]*featureCoefs['Height'],
					  arr[i+6]*featureCoefs['Slope']).getWave()
	return sample

def getError(arr):
	global signal
	#TO-DO: ADD CHECK AGAINST ALL FUNCTIONS
	sample = getWave(arr)
	signal = signal[:len(sample)]
	#Weight against all or nothing
	wAAON = 0
	#TO DO FINISH WEIGHTING 0 WEIGHTS! PROBABLY IF 0 NOT BEST DO IF SMALL
	for i in range(len(arr)):
		wAAON += (arr[0] - featureCoefs['Amplitude']/2.0)**4 +\
			   (arr[1] - featureCoefs['Pitch']/2.0)**4 +\
			   (arr[5] - featureCoefs['Height']/2.0)**4
	return (float(np.sum(abs(sample-signal))) + wAAON/(10**10), )

def info():
	top1 = tools.selBest(population, k=1)
	return top1[0]
def graph():
	top1 = tools.selBest(population, k=1)
	bestSine=getWave(top1[0])
	bestSine=bestSine/bestSine.max()
	plt.plot(signal)
	plt.plot(bestSine)
	plt.savefig('images/'+ str(generation) + '.png') 
	plt.clf()

def play():
	top1 = tools.selBest(population, k=1)
	bestSine=getWave(top1[0])
	bestSine=bestSine/bestSine.max()
	#PLAY AUDIO
	# for paFloat32 sample values must be in range [-1.0, 1.0]
	stream = p.open(format=pyaudio.paFloat32,
	                channels=1,
	                rate=Sound.fs,
	                output=True)
	stream.write(bestSine)
	stream.stop_stream()
	stream.close()
def saveAudio():
	global generation
	top1 = tools.selBest(population, k=1)
	bestSine=getWave(top1[0])
	bestSine=bestSine/bestSine.max()
	audioFile = wave.open('audio/' + str(generation)+'.wav', 'w')
	audioFile.setparams((2, 2, 11025, 0, 'NONE', 'not compressed'))
	for i in range(0, len(bestSine)):
		try:
			value = int(bestSine[i]*(2**15))
			if value > 32767:
				value = 32766
			elif value < -32767:
				value = -32766
			packed_value = struct.pack('h', value)
			audioFile.writeframes(packed_value)
			audioFile.writeframes(packed_value)
		except:
			pass

def main():
	global population, generation, record
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)
	toolbox = base.Toolbox()
	toolbox.register("daDouble", random.random)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.daDouble, n=numFeatures*numWaves)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("evaluate", getError)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=3)
	stats = tools.Statistics(key=lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)
	population = toolbox.population(n=5000)
	NGEN=20000
	try:
		for gen in range(NGEN):
			generation = gen
			key = daChar[0]
			if key != ' ':
				#Quit
				if key == 'q':
					break
				#Play
				elif key == 'p':
					play()
					daChar[0] = ' '
				#Info
				elif key == 'i':
					print(info())
					daChar[0] = ' '
				#Graph
				elif key == 'g':
					t = Thread(target=graph)
					t.start()
					# graph()
					daChar[0] = ' '
				#Record
				elif key == 'r':
					record = True
			if record == True:
				graph()
				saveAudio()
				with open("log.txt", "a") as f:
					f.write("Gen:" + str(gen) + '\n' + str(info()) + '\n\n')

			offspring = algorithms.varAnd(population, toolbox, cxpb=0.2, mutpb=0.3)
			fits = toolbox.map(toolbox.evaluate, offspring)
			for fit, ind in zip(fits, offspring):
				ind.fitness.values = fit
			population = toolbox.select(offspring, k=len(population))
			print(chr(27) + "[2J")
			logs = stats.compile(population)
			print("Gen:", gen, "\n", logs)
	except KeyboardInterrupt:
		pass
	p.terminate()

t = Thread(target=main)
t.start()
checkInput()