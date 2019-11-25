
"""Tools for Vector Arithmetic."""


import random
import re
import sys

from math import pow, sqrt, floor, inf, exp


__name__ = "Vectortools"
__version__ = "0.2.0"
__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


class Vector():

	def __init__(self, *args):

		if len(args) == 1:
			param = args[0]
			if type(param) in (list, tuple, range, set):
				args = param
			elif type(param)==dict:
				args = param.values()
			elif type(param) in (int, float):
				pass
			else:
				raise TypeError(
					"Type passed to Vector must be "
					"list, tuple, dict, int or float"
				)

		args = list(args)
		mincomp = (inf, inf)
		maxcomp = (-inf, inf)
		vecsum = 0
		
		self.dimensions = 0
		
		for index in range(len(args)):
			val = args[index]
			try:
				val = float(val)
				args[index] = val

				if val < mincomp[0]:
					mincomp = val, index
				elif val > maxcomp[0]:
					maxcomp = val, index
				else:
					pass

				vecsum += val
				self.dimensions += 1

			except TypeError:
				raise TypeError(
					"Value '{}' is not a number. "
					"Vector components must be numbers".format(val)
				)

		self.components = list(args)


		# Force minimum vector length to be 2
		# if there's only one component add 0 as second component
		if self.dimensions == 1:
			mincomp = (0, 1)
			self.pad(2, 0)

		# prestore queries for reusability, 
		# to avoid needless time complexities
		self.min = mincomp
		self.max = maxcomp
		self.sum = vecsum
		self.memsize = self.__memsize__()

	def __add__(self, other):
		if isinstance(other, Vector):
			a, b = self.__dress__(other)
			magnitudes = [sum(pair) for pair in zip(a, b)]
			result = Vector(magnitudes)

			return result
		else:
			raise TypeError(
				"Addends must be of the same <Vector> type"
			)

	def __isvector__(self, other):		
		return isinstance(other, Vector)
		
	def __contains__(self, number):
		"""Checks if a given number is a component of a Vector.
		
		Arguments
		---------
		:number: the number which component membership is to be checked.
		"""
		return True if number in self.components else False

	def __covcalcs__(self, other):
		"""Returns the covariance between two vectors, and 
		their respective standard deviations.

                Arguments
		----------
		:other: a Vector
		"""
		
		self.__isequaldim__(other)
		
		x = self.components
		y = other.components
		x_ = self.mean()
		y_ = other.mean()
		x_std = self.std()
		y_std = other.std()
		n = self.dimensions
		sum_mean_diff = 0

		for index in range(n):
			sum_mean_diff += (x[index] - x_) * (y[index] - y_)
		
		return (
			sum_mean_diff / n,
			x_std,
			y_std
		)

	def __describe__(self):
		"""Returns a statement of the Vector's dimensions"""

		components_str = ", ".join(
			[str(x) for x in self.components]
		)

		return (
			"A {}-dimensional vector. "
			"[ Memory Size {} bytes ]".format(
				self.dimensions, self.memsize
			)
		)

	def __dimensions__(self):
		"""Returns an integer representing a Vector's number of dimensions"""

		return len(self.components)

	def __dress__(self, other, extension_component=0):
		"""Compares Vectors and pads the Vector with lowest number of dimensions with a given component.

		Arguments
		---------
		other : the other Vector

		extension_component: the number to use to pad the Vector 
				     with lowest dimensions. Defaults to 0.
		"""

		if self.dimensions < other.dimensions:
			return(
				self.padded(other.dimensions, extension_component).components,
				other.components
			)
		elif other.dimensions < self.dimensions:
			return(
				self.components,
				other.padded(self.dimensions, extension_component).components
			)
		else:
			return self.components, other.components
	
	def __errdiff__(self, other, absolute=False):
		
		if not isinstance(other, Vector):
			raise TypeError(
				"This requires two Vectors. A {} "
				"is not valid".format(type(other))
			)
		
		# Vectors must be of same length
		if self.dimensions==other.dimensions:
			diff = []
			for index in range(self.dimensions):
				error_ = self.components[index] - other.components[index]

				if absolute:
					# if the absolute parameter is set to True,
					# an absolute value of the error is returned,
					# instead of the raw value
					error_ = abs(error_)

				diff.append(error_)
			
			return diff
		else:
			raise ValueError(
				"Vectors must be of the same length"
			)
		
	def __eq__(self, other):
		if not isinstance(other, Vector):
			return False

		return self.components==other.components

	def __getitem__(self, key):
		return self.components[key]

	def __hash__(self):
		return hash(self.components)

	def __isequaldim__(self, other):
		"""Checks if the vectors have equal dimensions."""

		if self.dimensions != other.dimensions:
			raise ValueError(
				"Vectors do not have the same dimensions"
			)

	def __len__(self):
		return self.dimensions

	def __mean__(self):
		"""Returns the mean of the components of a Vector"""

		return self.sum/self.dimensions

	def __memsize__(self, stringformat=False):
		"""Returns the memory size of a Vector in bytes.
		
		Arguments:
		----------
		:stringformat: (Boolean) specifies whether memory size should be 
		              returned as a string. Default is False.
		
		"""

		props = (
			self,
			self.components,
			self.dimensions,
			self.min,
			self.max,
			self.sum
		)

		memsize = 0

		for prop in props:
			memsize += sys.getsizeof(prop)

		return (
			"{} bytes".format(memsize) if stringformat else memsize
		)

	def __mul__(self, other):
		if type(other) in (int, float):
			return self.smul(other)
		elif isinstance(other, Vector):
			return self.dotmul(other)
		else:
			raise TypeError(
				"Type mismatch! You cannot multiply"
				" a <Vector> and {}.".format(type(other))
			)

	def __ne__(self, other):
		return self.components!=other.components

	def __neg__(self):
		return self.reversed()

	def __pow__(self, power):
		return Vector(
			[pow(X, power) for X in self.components]
		)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __repr__(self, matrixmode=False):
		if self.dimensions <= 12:
			value_str = " ".join(
				[str(round(val, 3)) for val in self.components]
			)
		else:
			head =  " ".join(
				[str(round(val, 3)) for val in self.components[:5]]
			)
			
			tail =  " ".join(
				[str(round(val, 3)) for val in self.components[-5:]]
			)
			
			value_str = "...".join((head, tail))

		return "Vector({})".format(value_str) if not matrixmode else value_str

	def __setitem__(self, index, value):
		if type(value) in (int, float):
			replaced_val = self.components[index]

			self.components[index] = value

			self.sum = self.sum - replaced_val + value
			if self.min[0] > value: self.min = value, index
			if self.max[0] < value: self.max = value, index
		else:
			raise TypeError(
				"Value to be inserted must be a number"
			)
			
	def __std__(self):
		"""Returns the standard deviation of the components of a Vector"""

		mean = self.__mean__()
		component_count = self.dimensions
		
		sum_sq_diff = 0
		
		for x in self.components:
			sum_sq_diff += pow(x - mean, 2)
		
		return sqrt(
			sum_sq_diff/component_count
		)

	def __str__(self):
		return self.__repr__()

	def __sub__(self, other):
		if isinstance(other, Vector):
			a, b = self.__dress__(other)
			magnitudes = [x - y for x, y in zip(a, b)]
			result = Vector(magnitudes)

			return result
		else:
			raise TypeError(
				"Operands must be of the same <Vector> type"
			)

	def add(self, other):
		"""Adds two Vectors.
		
		Arguments
		----------
		:other: a Vector
		"""

		return self.__add__(other)

	def append(self, value):
		"""Appends a component or a sequence of components to a Vector.
		
		Arguments
		----------
		:value: component to be added to a Vector. 
		        A valid value must be of type int or float.
		"""

		def updateminmaxsum(value, index_):
			# update the minimum, maximum components, and sum 
			# after appending new value(s)
			self.sum += value
			self.dimensions += 1

			if self.min[0] > value:
				self.min = value, index_

			if self.max[0] < value:
				self.max = value, index_
		
		index_ = self.dimensions
		
		if type(value) in (int, float):
			value = float(value)			
			self.components.append(value)
			updateminmaxsum(value, index_)
			
		elif type(value) in (tuple, list, set, range):
			index_ = self.dimensions
			for i in value:
				if type(i) in (int, float):
					if type(i)==int:
						i = float(i)

					self.components.append(i)
					updateminmaxsum(i, index_)
					index_ += 1
				else:
					raise TypeError(
						"Argument must be an integer or float"
					)
		else:
			raise TypeError(
				"Argument must be an integer, float, tuple, list"
			)

	def concat(self, other):
		"""Concatenate two Vectors.
		
		Aurguments
		----------
		:other: a Vector		
		"""

		if isinstance(other, Vector):
			return Vector(self.components + other.components)
		else:
			raise TypeError(
				"concat requires only <Vector> types"
			)

	def cov(self, other):
		"""Returns the covariance between two vectors.
		
		Arguments
		---------
		:other: A vector
		"""
		
		cov_, _, _ = self.__covcalcs__(other)
		
		return cov_

	def corr(self, other):
		"""Returns the correlation between two vectors.
		
		Aurguments
		----------
		:other: a Vector
		"""
		
		cov_, x_std, y_std = self.__covcalcs__(other)
		
		
		return cov_ / (x_std * y_std)

	def cosinesim(self, other):
		"""Returns the cosine similarity between two vectors
		
		Arguments
		---------
		:other: A vector
		"""
		
		# check if vectors have equal dimensions
		self.__isequaldim__(other)
		
		dot = self.dotmul(other)
		mag_product = self.magnitude() * other.magnitude()

		return dot/mag_product

	def cost(self, other):
		"""The cost function on two vectors, assuming the second vector 
		compared to the first vector which is the groud truth.

		Arguments
		---------
		:other: A vector
		"""
		
		# check if vectors have equal dimensions
		self.__isequaldim__(other)
		
		diffs = self.__errdiff__(other)
		
		sum_diffs = 0
		for diff in diffs:
			sum_diffs += pow(diff, 2)
			
		return sum_diffs/(2 * self.dimensions)

	def crossmul(self, other):
		"""Returns the cross product of two vectors in 3-D space.
		
		Arguments
		---------
		:other: A vector
		"""

		if isinstance(other, Vector):
			if self.dimensions==other.dimensions==3:
				a1, a2, a3 = self.components
				b1, b2, b3 = other.components

				result = Vector(
					a2*b3 - a3*b2,
					a3*b1 - a1*b3,
					a1*b2 - a2*b1
				)

				return result
			else:
				raise ValueError(
					"Vectors must be 3-dimensional"
				)
		else:
			raise ValueError(
				"crossmul() requires 3-dimensional "
				"<Vector> types"
			)

	def describe(self):
		"""Describes a Vector in words."""
		
		return self.__describe__()

	def distance(self, other):
		"""Returns the euclidean distance between two Vectors.

		Arguments
		----------
		:other: a Vector

		"""

		if isinstance(other, Vector):
			a, b = self.__dress__(other)
			
			sum_sq_diff = 0
			
			for x, y in zip(a, b):
				sum_sq_diff += pow((x - y), 2)

			return sqrt(sum_sq_diff)
		else:
			raise TypeError(
				"distance() requires <Vectors> types"
			)

	def dotmul(self, other):
		"""Returns the dot product of two vectors.

		Arguments
		---------
		:other: a Vector
		"""

		if isinstance(other, Vector):
			a, b = self.__dress__(other)

			dmul = 0
			for x, y in zip(a, b):
				dmul += x * y 

			return dmul
		else:
			raise TypeError(
				"dotmul() requires only <Vector> types"
			)

	def ediv(self, other):
		"""Performs element-wise division on two vectors.
		
		Aurguments
		----------
		:other: a Vector
		
		"""
		
		# check if vectors have equal dimensions
		self.__isequaldim__(other)
		
		self_comps = self.components
		other_comps = other.components
		
		return Vector(
			[x / y for x, y in zip(self_comps, other_comps)]
		)
		
	def emul(self, other):
		"""Performs element-wise multiplication on two vectors
		
		
		
		"""
		
		# check if vectors have equal dimensions
		self.__isequaldim__(other)
		
		self_comps = self.components
		other_comps = other.components
		
		return Vector(
			[x * y for x, y in zip(self_comps, other_comps)]
		)

	def insert(self, index, value):
		"""Inserts a new component.

		Arguments
		---------
		index : Index where new component must be inserted

		value : Component to be inserted

		"""

		if type(value) in (int, float):
			self.components.insert(index, value)
			self.dimensions += 1
			
			self.sum = self.sum + value
			if self.min[0] > value:
				self.min = value, index
				if self.max[1] >= index:
					self.max = self.max[0], self.max[1]+1

			if self.max[0] < value:
				self.max = value, index
				if self.min[1] >= index:
					self.min = self.min[0], self.min[1]+1
		else:
			raise TypeError(
				"value must be an integer or float"
			)

	def jaccard(self, other):
		"""Returns the jaccard similarity between two vectors"""
		
		jindex = 0
		
		if self.dimensions==other.dimensions:
			for x, y in zip(self.components, other.components):
				if x==y:
					jindex += 1
		else:
			raise ValueError(
				"Vectors must be of the same length"
			)

		return round(jindex/self.dimensions, 2)
	
	def join(self, separator=" "):
		"""Returns a string which is a concatenation of components of a vector"""

		return " ".join(
			[str(val) for val in self.components]
		)

	def leakyrelu(self):
		"""Passes a vector through the Leaky Rectified Linear Unit Function."""

		return self.relu(0.01)

	def leastdev(self, other):
		"""Returns the Least Absolute Deviation(L1 Norm) between two vectors"""

		diffs = self.__errdiff__(other, absolute=True)
		return sum(diffs)

	def leastsq(self, other):
		"""Returns the Least Squares(L2 Norm) between two vectors"""

		diffs = self.__errdiff__(other)
		
		sum_diffs = 0
		for diff in diffs:
			sum_diffs += pow(diff, 2)
		
		return sum_diffs

	def magnitude(self):
		"""Returns the magnitude of a Vector"""

		sum_sq_mul = 0
		for x in self.components:
			sum_sq_mul += pow(x, 2)

		return sqrt(sum_sq_mul)

	def mean(self):
		"""Returns the mean of the components of a Vector."""

		return self.__mean__()

	def minmax(self, a=0, b=1):
		"""Normalizes using min-max feature scaling.
		
		Arguments
		---------
		a : minimum value of the scaling range
		b : maximum value of the scaling range
		
		"""

		min_ = self.min[0]
		max_ = self.max[0]
		
		if a==0 and b==1:
			norm_val = lambda X: (
				(X - min_)/(max_ - min_)
			)
		else:
			norm_val = lambda X: (
				a + (((X - min_)*(b - a)) /(max_ - min_))
			)

		return Vector([
			norm_val(X) for X in self.components
		])

	def minmaxmean(self):
		"""Normalizes vector using the mean feature scaling."""

		mean = self.__mean__()

		return Vector([
			(X - mean)/(self.max[0] - self.min[0]) for X in self.components
		])

	def mae(self, other):
		"""Returns the Mean Absolute Error between two vectors.
		
		Arguments
		----------
		:other: a Vector
		"""
		
		diffs = self.__errdiff__(other, absolute=True)
		return sum(diffs)/self.dimensions
		
	def mbe(self, other):
		"""Returns the Mean Bias Error between two vectors.
		
		Arguments
		----------
		:other: a Vector
		"""
		
		diffs = self.__errdiff__(other)
		return sum(diffs)/self.dimensions
		
	def mse(self, other):
		"""Returns the Mean Square Error between two vectors.
		
		Arguments
		----------
		:other: a Vector
		"""
		
		diffs = self.__errdiff__(other)
		
		sum_diffs = 0
		for diff in diffs:
			sum_diffs += pow(diff, 2)
			
		return sum_diffs/self.dimensions

	def normalize(self, mode="zscore"):
		"""Returns a normalized variant of this vector using a specified method.

		Parameters
		----------
		mode: The method to be used. Default is "zcore"
			"zcore" - uses z-core feature scaling
			"minmax" - uses the minimum and maximum feature scaling
			"minmaxmean" - uses the mean, minimum and maximum feature scaling
		"""
		
		if mode not in ("zscore", "minmax", "minmaxmean"):
			raise ValueError(
				"{} is not a valid mode".format(mode)
			)
		
		modes = {
			"zscore":self.stdnorm(),
			"minmax":self.minmax(),
			"minmaxmean":self.minmaxmean()
		}
		
		return modes.get(mode)

	def pad(self, desired_length, extension_component=0):
		"""Extends a Vector with several of a specified component.

		Parameters
		----------
		desired_length : The expected number of dimensions for the padded Vector

		extension_component : The number to use to pad the Vector. 
					 Defaults to 0.

		"""

		self.components = self.padded(
			desired_length, extension_component
		).components

		self.dimensions = self.__dimensions__()

	def padded(self, desired_length, extension_component=0):
		"""Returns a new Vector, which is the original Vector padded with several of a specified component.

		Example
		-------
		if x = Vector(2,3),
		x.padded(4, 1) returns Vector(2.0 3.0 1.0 1.0)

		*** x is preserved as a new Vector is returned.

		Parameters
		----------
		desired_length : The expected number of dimensions for the padded Vector

		extension_component : The number to use to pad the Vector. Defaults to 0.
		
		"""

		if (
			type(desired_length)==type(extension_component)==int and
			desired_length > 0
		):
			deficit = desired_length - len(self)
			pad_list = [extension_component for i in range(deficit)]
			ext_vector = Vector(self.components + pad_list)

			return ext_vector
		else:
			raise ValueError(
				"All arguments to padded() should be valid positive integers"
			)

	def pararelu(self, scalefactor):
		"""Passes a vector through the Parametric Rectified Linear Unit Function.
		
		Arguments
		----------
		:scalefactor: scaling factor for negative components a vector
		"""

		return self.relu(scalefactor)

	def pop(self, index=None):
		"""Deletes a component.

		Parameters
		----------
		index : Index of component to be deleted.
				Defaults to the index of the last component.

		"""

		if  (type(index)==int and index >= 0) or index==None:
			index = index if index != None else self.dimensions - 1

			val = self.components.pop(index)
			self.sum = self.sum - val
			self.dimensions -= 1

			if val in (self.min[0], self.max[0]):
				sorted_comp = sorted(self.components)
				new_min = sorted_comp[0]
				self.min = new_min, self.components.index(new_min)
				new_max = sorted_comp[-1]
				self.max = new_max, self.components.index(new_max)

		else:
			raise ValueError(
				"index argument must be a positive integer"
			)
	
	def pow(self, exponent):
		"""Raises each component of a Vector to given power.
		
		Arguments:
		----------
		:exponent: the power components should be raised to.
		"""
		
		return self.__pow__(exponent)

	def relu(self, coef=0):
		"""Passes a vector through Rectified Linear Unit Function.
		
		Parameters
		----------
		
		coef - a value to be used for changing between standard relu(0), 
		       leaky relu(0.01), and parametric relu.
		
		"""

		relu = lambda y: max(coef * y, y)

		return Vector(
			[relu(y) for y in self.components]
		)

	def rmse(self, other):
		"""Returns the Root Mean Square Error between two vectors.
		
		Arguments:
		----------
		:other: a Vector
		"""
		
		return sqrt(self.mse(other))
		
	def rsquare(self, other):
		"""Returns the R-Square Score for two compared vectors.
		
		Arguments:
		----------
		:other: a Vector
		"""
		
		# check if vectors have equal dimensions
		self.__isequaldim__(other)

		mean = self.mean()
		sum_sq_mean_diff = 0

		for index in range(self.dimensions):
			error_ = self.components[index] - mean
			sum_sq_mean_diff += pow(error_, 2)

		return 1 - (self.leastsq(other)/sum_sq_mean_diff)

	def reverse(self):
		"""Sets a Vector in the opposite direction."""

		self.components = self.reversed().components

	def reversed(self):
		"""Returns a new Vector whose direction is opposite that of the original Vector.

		Unlike reverse(), this preserves original Vector.
		"""

		r_vec = self.smul(-1)		
		return r_vec

	def sdiv(self, scalar):
		"""Returns a Vector, which is the result of a scalar division.

		Parameters
		----------
		scalar: a number of int or float type to divide the vector.
		
		"""

		if type(scalar) in (int, float):
			return Vector([
				component/scalar for component in self.components
			])
		else:
			raise TypeError(
				"Second other must be a scalar"
			)

	def shuffle(self):
		"""Shuffles the components of this Vector in-place."""

		random.shuffle(self.components)

	def shuffled(self):
		"""Returns a Vector whose components are shuffled version of this vector."""

		return Vector(
			random.sample(self.components, self.dimensions)
		)

	def sigmoid(self):
		"""Passes vector through a sigmoid function and returns a new vector."""

		sig = lambda x: 1 / (1 + exp(-x))

		return Vector([
			sig(y) for y in self.components
		])

	def smul(self, multiplier):
		"""Returns the product of a scalar multiplication.

		Parameters
		----------
		:multiplier: a number (of int or float type) to multiply through components of a Vector
		"""

		if type(multiplier) in (int, float):
			return Vector([
				component * multiplier for component in self.components
			])
		else:
			raise TypeError(
				"The multiplier must be a scalar"
			)

	def softmax(self):
		"""Passes vector through a softmax function and returns a new vector"""

		exps = []
		sum_exp = sum(exps)
		
		for x in self.components:
			exp_x = exp(x)
			sum_exp += exp_x
			exps.append(exp_x)

		soft = lambda y: y / sum_exp

		return Vector([
			soft(p) for p in exps
		])

	def std(self):
		"""Returns the standard deviation of the components of a Vector."""

		return self.__std__()

	def stdnorm(self):
		"""Returns a normalised variant of this Vector."""

		mean = self.__mean__()

		# Could have used already computed standard deviation
		# but that would not reuse the mean and would increase
		# time complexity.
		std = sqrt(
			sum([pow(x - mean, 2) for x in self.components])/self.dimensions
		)

		if std > 0:
			return Vector(
				[(x - mean)/std for x in self.components]
			)
		else:
			if self.components[0] <= 1 and self.components[0] >= -1:
				# if components are the same and within range -1 to 1
				return self
			else:
				# if components are the same and not within range -1 to 1
				return Vector(
					[1 for _ in range(self.dimensions)]
				)

	def subtract(self, other):
		"""Performs Vector subtraction.

		Parameters
		----------
		other : a Vector
		"""

		return self.__sub__(other)


	def subvec(self, start, end):
		"""Creates a new Vector from a sequence of components from another Vector.

		Parameters
		----------
		:start : the beggining index of the selected sub-sequence

		:end : the ending index of the selected sub-sequence
		"""

		return Vector(self.components[start:end])

	def tanh(self):
		"""Passes vector through the tanh function and returns new vector."""

		tnh = lambda y: (2 / (1 + exp(-2 * y))) - 1

		return Vector(
			[tnh(y) for y in self.components]
		)

	def to_list(self):
		"""Returns a list of a Vector's components."""

		return self.components

	def to_str(self):
		"""Returns a string representation of a Vector."""

		return self.__repr__()

	def to_tuple(self):
		"""Returns a string representation of a Vector."""

		return tuple(self.components)

	def unitvec(self):
		"""Returns a normalized variant of this Vector, scaled to unit length."""
	
		magnitude = self.magnitude()
		
		return (
			Vector([x/magnitude for x in self.components])
		)

	def vector_eq(self, other):
		"""Returns the vector equation of a line between two vectors"""
	
		return (
			"eq = [{}] + t[{}]".format(
				self.__repr__(True), other.subtract(self).__repr__(True)
			)
		)
	
	@staticmethod
	def flatten(args):
		"""Converts a sequence of valid types into a single Vector.
		
		Arguments
		---------
		args: A list, tuple, or set of items. Items may be numbers, or a valid sequence like lists, 
		      tuples, sets, vectors, containing only numbers.
		"""

		new_components = []

		for arg in args:
			if isinstance(arg, Vector):
				new_components += arg.components
			elif isinstance(arg, list):
				new_components += arg
			elif isinstance(arg, (int, float)):
				new_components.append(arg)
			elif isinstance(arg, (tuple, range, set)):
				new_components += list(arg)
			else:
				raise TypeError(
					"Arguments must be any of lists, tuples, sets, vectors, or numbers"
				)
		
		return Vector(new_components)
	
	@staticmethod
	def sum(args):
		"""Add multiple Vectors.
		
		Arguments
		---------
		args: a list, tuple, or set of Vectors
		"""

		vecsum = args[0]
		args = args[1:]

		for arg in args:
			if isinstance(arg, Vector):
				vecsum = vecsum.add(arg)
			else:
				raise TypeError(
					"Arguments must be any of lists, tuples, sets, vectors, or numbers"
				)
		
		return vecsum
				
				
		 

def isovector(component, dimension):
	"""Create a homogenous Vector of a specified dimension.

	Parameters
	----------
	component : The desired component.

	dimension : The desired dimensions of the Vector to be created.

	"""

	if type(component) in (int, float):
		if type(dimension)==int:
			pass
		elif type(dimension)==float:
			dimension = floor(dimension)
		else:
			raise ValueError(
				"second argument <dimension> should be a positive integer"
			)

		return Vector(
			[component for i in range(dimension)]
		)
	else:
		raise TypeError(
			"first argument <number> must be a number"
		)

def randvec(dimensions, seed=None):
	"""Generates a random Vector.
	
	Parameters:
	-----------
	
	dimensions : the number of components of the resultant vector.

	seed : a seed value for generation of numbers. Default is None.
	
	"""
	
	if seed==None:
		pass
	else:
		random.seed(seed)
	
	return Vector(
		random.sample(
			range(-dimensions, dimensions), dimensions
		)
	)

def main():

	interactive_shell_header = (
		"====================================================="
		"\n\n"
		"\tVECTORKIT :: v{} \n"
		"\tInteractive Shell"
		"\n\n"
		"====================================================="
	).format(__version__)

	code_template = (
		"_______________ = {}\n"
		"if _______________ or _______________ == False:\n"
		"\tprint(_______________)\n"		
	)

	print(interactive_shell_header)

	line = input(">>>> ")

	while line!="exit()":		
		try:
			if re.match(r'^print\s|^[^=\s]+\s= ', line):
				exec(line)
			elif line.strip()=="":
				pass
			else:
				exec(code_template.format(line.strip()))
		except Exception as e:
			print("Error! ", e.args)	

		line = input(">>>> ")
