
"""Tools for Vector Arithmetic."""


import random
import re
import sys

from math import pow, sqrt, floor, inf, exp


__name__ = "Vectortools"
__version__ = "0.1.6"
__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


class Vector():

	def __init__(self, *args):

		if len(args) == 1:
			param = args[0]
			if type(param) in (list, tuple, range):
				args = param
			elif type(param)==dict:
				args = list(param.values())
			elif type(param) in (int, float):
				pass
			else:
				raise ValueError(
					"Type passed to Vector must be "
					"list, tuple, dict, int or float"
				)

		args = list(args)
		mincomp = (inf, inf)
		maxcomp = (-inf, inf)
		vecsum = 0
		
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

			except ValueError:
				raise ValueError(
					"Value '{}' is not a number. "
					"Vector components must be numbers".format(val)
				)

		self.components = list(args)
		self.dimensions = self.__dimensions__()

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
			result = Vector(*magnitudes)

			return result
		else:
			raise TypeError(
				"Addends must be of the same <Vector> type"
			)

	def __contains__(self, number):
		return True if number in self.components else False

	def __covcalcs__(self, other):
		"""Returns the covariance between two vectors, and 
		their respective standard deviations.
		
		"""

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
		"""Returns a statement of the Vector's dimension."""

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
		"""Returns an integer representing a Vector's number of dimensions."""

		return len(self.components)

	def __dress__(self, other, extension_component=0):
		"""Compares Vectors and pads the Vector with lowest number of dimensions.

		Parameters
		----------
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

	def __eq__(self, other):
		return self.components==other.components

	def __getitem__(self, key):
		return self.components[key]

	def __hash__(self):
		return hash(self.components)

	def __len__(self):
		return self.dimensions

	def __mean__(self):
		"""Returns the mean of the components of a Vector"""

		return self.sum/self.dimensions

	def __memsize__(self, stringformat=False):
		"""Returns the memory size of a Vector in bytes.
		
		Arguments:
		
		stringformat: (Boolean) specifies whether memory size should be 
		              returned as a string
		
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
			raise ValueError(
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
		"""Returns the standard deviation of the components of a Vector."""

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
		"""Adds two Vectors."""

		return self.__add__(other)

	def append(self, value):
		"""Appends a component to a Vector."""

		if type(value) in (int, float):
			value = float(value)			
			self.components.append(value)
			
			self.sum += value
			
			index = self.dimensions
			if self.min[0] > value: self.min = value, index
			if self.max[0] < value: self.max = value, index
			
		elif type(value) in (tuple, list):
			for i in value:
				if type(i) in (int, float): 
					self.append(float(i))
				else:
					raise ValueError(
						"Argument must be an integer or float"
					)
		else:
			raise ValueError(
				"Argument must be an integer, float, tuple, list"
			)

		self.dimensions = self.__dimensions__()

	def concat(self, other):
		"""Concatenate two Vectors."""

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
		
		Other: the another vector
		"""
		
		cov_, _, _ = self.__covcalcs__(other)
		
		return cov_

	def corr(self, other):
		"""Returns the correlation between two vectors."""
		
		cov_, x_std, y_std = self.__covcalcs__(other)
		
		
		return cov_ / (x_std * y_std)

	def crossmul(self, other):
		"""Returns the cross product of two vectors in 3-D space."""

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

		Aurguments
		----------
		other : the other Vector

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
		
		other : a Vector
		"""

		if isinstance(other, Vector):
			a, b = self.__dress__(other)

			dmul = 0
			for x, y in zip(a, b):
				dmul += x * y 

			return dmul
		else:
			raise ValueError(
				"dotmul() requires only <Vector> types"
			)

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

	def insert(self, index, value):
		"""Inserts a new component.

		Arguments
		---------
		index : Index where new component must be inserted

		value : Component to be inserted

		"""

		if type(value) in (int, float):
			self.components.insert(index, value)
			self.dimensions = self.__dimensions__()
			
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

	def magnitude(self):
		"""Returns the magnitude of a Vector."""

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
		"""Normalizes vector using the mean."""

		X_ = self.__mean__()

		return Vector([
			(X - X_)/(self.max[0] - self.min[0]) for X in self.components
		])

	def mse(self, other):
		"""Returns the Mean Square Error between two vectors."""

		# Vectors must be of same length
		if self.dimensions==other.dimensions:
			SE = 0
			for index in range(self.dimensions):
				SE += (self.dimensions[index] - other.dimensions[index])**2
			
			return SE/self.dimensions
		
		else:
			raise ValueError(
				"Vectors must be of the same length"
			)
				

	def normalized(self):
		return self.stdnorm()

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
			self.dimensions = self.__dimensions__()

			if val in (self.min[0], self.max[0]):
				sorted_comp = sorted(self.components)
				new_min = sorted_comp[0]
				self.min = new_min, self.components.index(new_min)
				new_max = sorted_comp[-1]
				self.max = new_max, self.components.index(new_max)

		else:
			raise ValueError(
				"Index argument must be a positive integer"
			)

	def relu(self):
		"""Passes a vector through Rectified Linear Unit Function."""

		relu = lambda y: max(0, y)

		return Vector(
			[relu(y) for y in self.components]
		)

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
			raise ValueError(
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

	def smul(self, scalar):
		"""Returns the product of a scalar multiplication.

		Parameters
		----------
		scalar: a number (of int or float type) to multiply vector.

		"""

		if type(scalar) in (int, float):
			return Vector([
				component * scalar for component in self.components
			])
		else:
			raise ValueError(
				"Second other must be a scalar"
			)

	def softmax(self):
		"""Passes vector through a softmax function and returns a new vector."""

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
			if self.components[0] in (-1, 0, 1):
				return self
			else:
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
		start : the beggining index of the selected sequence

		end : the ending index of the selected sequence

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
		"""Returns the vectpr equation of a line between two vectors"""
	
		return (
			"eq = [{}] + t[{}]".format(
				self.__repr__(True), other.subtract(self).__repr__(True)
			)
		)
		 

def isovector(component, dimension):
	"""Create a homogenous Vector of a specified dimension.

	Example
	-------
	isovector(2, 4) returns Vector(2.0 2.0 2.0 2.0)

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
		raise ValueError(
			"first argument <number> must be a number"
		)

def randvec(dimensions):
	"""Generates a random Vector.
	
	Parameters:
	-----------
	
	dimensions : the number of components of the resultant vector
	"""
	
	return Vector(
		random.sample(
			range(-dimensions, dimensions), dimensions
		)
	)

def main():

	interactive_shell_header = (
		"====================================================="
		"\n\n"
		"\tVECTORKIT :: v0.1.6 \n"
		"\tInteractive Shell"
		"\n\n"
		"====================================================="
	)

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