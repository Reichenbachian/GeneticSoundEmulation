import numpy as np
import math
class Sound:
	fs = 11025       # sampling rate, Hz, must be integer
	def __init__(self, _volume, _f, decay, daType, shift, height, slope, _duration=2.806984127):
		if _volume > 1:
			_volume = 1
		elif _volume < 0:
			_volume = 0
		self.volume = _volume
		self.f = int(_f)
		self.duration = _duration
		self.decay = decay
		self.daType = daType
		self.shift = shift
		self.height = int(height)
		self.slope = slope
	def __add__(self, other):
		return self.getSine() + other.getSine()
	def getWave(self):
		domain = np.arange(1+self.shift, self.fs*self.duration+1+self.shift)
		#sin
		if self.daType <= .3:
			if self.decay == 0:
				return (np.sin(2*np.pi*domain.copy()*self.f/self.fs)).astype(np.float32)
			return (math.e**(-1*domain.copy()/self.decay)*np.sin(2*np.pi*domain.copy()*self.f/self.fs)).astype(np.float32)
		#cos
		if self.daType <= .6:
			if self.decay == 0:
				return (np.cos(2*np.pi*domain.copy()*self.f/self.fs)).astype(np.float32)
			return (math.e**(-1*domain.copy()/self.decay)*np.cos(2*np.pi*domain.copy()*self.f/self.fs)).astype(np.float32)
		#square
		if self.height == 0:
			return domain*0
		if self.decay == 0:
			return ((self.slope*domain)%self.height - self.height/2).astype(np.float32)
		return (math.e**(-1*domain.copy()/self.decay)*(self.slope*domain%self.height) - self.height/2).astype(np.float32)
	def __str__(self):
		return "Volume: " + str(self.volume) + " Freq: " + str(self.f) + " Volume: " + str(self.volume)