
"""Tools for Vector Arithmetic."""


from math import pow, sqrt, floor
import re


__name__ = "Vectortools"
__version__ = "0.1.3"
__author__ = "Victor Mawusi Ayi"


class Vector():

	def __init__(self, *args):

		if len(args) == 1:
			param = args[0]
			if type(param) in (list, tuple):
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

		for index in range(len(args)):
			val = args[index]
			try:
				args[index] = float(val)
			except ValueError:
				raise ValueError(
					"Value '{}' is not a number. "
					"Vector components must be numbers".format(val)
				)

		self.components = list(args)
		self.dimensions = self.__dimensions__()

		if self.dimensions == 1:
			self.extend(2, 0)

	def __add__(self, operand):
		if isinstance(operand, Vector):
			a, b = self.__dress__(operand)
			magnitudes = [sum(pair) for pair in zip(a, b)]
			result = Vector(*magnitudes)

			return result
		else:
			raise TypeError(
				"Addends must be of the same <Vector> type"
			)

	def __contains__(self, number):	
		return True if number in self.components else False

	def __describe__(self):
		"""Returns a statement of the Vector's dimension."""

		components_str = ", ".join([str(x) for x in self.components])
		return (
			"A {}-dimensional vector with components: "
			"{}".format(self.dimensions, components_str)
		)

	def __dimensions__(self):
		"""Returns an integer representing a Vector's number of dimensions."""

		return len(self.components)

	def __dress__(self, operand, extension_component=0):
		"""Compares Vectors and extends the Vector with lowest number of dimensions.

		Parameters
		----------
		operand : the operand Vector

		extension_component : the number to use to extend the Vector with lowest dimensions. 
					 Defaults to 0.
		"""

		if self.dimensions < operand.dimensions:
			return(
				self.extended(operand.dimensions, extension_component).components,
				operand.components
			)
		elif operand.dimensions < self.dimensions:
			return(
				self.components,
				operand.extended(self.dimensions, extension_component).components
			)
		else:
			return self.components, operand.components

	def __eq__(self, other):
		return self.components==other.components

	def __getitem__(self, key):
		return self.components[key]

	def __hash__(self):
		return hash(self.components)

	def __len__(self):
		return self.dimensions

	def __mul__(self, operand):
		if type(operand) in (int, float):
			return self.smul(operand)
		elif isinstance(operand, Vector):
			return self.dotmul(operand)
		else:
			raise ValueError(
				"Type mismatch! You cannot multiply"
				" a <Vector> and {}.".format(type(operand))
			)

	def __rmul__(self, operand):
		return self.__mul__(operand)

	def __ne__(self, operand):
		return self.components!=operand.components

	def __repr__(self):
		value_str = " ".join(
			[str(val) for val in self.components]
		)
		return "Vector({})".format(value_str)

	def __setitem__(self, key, value):
		if type(value) in (int, float):
			self.components[key] = value		
		else:
			raise TypeError(
				"Value to be inserted must be a number"
			)

	def __str__(self):
		return self.__repr__()

	def __sub__(self, operand):
		if isinstance(operand, Vector):
			a, b = self.__dress__(operand)
			magnitudes = [x - y for x, y in zip(a, b)]
			result = Vector(*magnitudes)

			return result
		else:
			raise TypeError(
				"Operands must be of the same <Vector> type"
			)

	def add(self, operand):
		"""Adds two Vectors."""

		return self.__add__(operand)

	def append(self, value):
		"""Appends a component to a Vector."""

		if type(value) in (int, float):
			self.components.append(value)
		elif type(value)==list:
			self.components = self.components + value
		elif type(value)==tuple:
			self.components = self.components + list(value)
		else:
			raise ValueError(
				"argument must be an integer, float, tuple, or list"
			)

		self.dimensions = self.__dimensions__()

	def concat(self, operand):
		"""Concatenate two Vectors."""

		if isinstance(operand, Vector):
			return Vector(self.components + operand.components)
		else:
			raise TypeError(
				"concat requires only <Vector> types"
			)

	def crossmul(self, operand):
		"""Returns the cross product of two vectors in 3-D space."""

		if isinstance(operand, Vector):
			if self.dimensions==operand.dimensions==3:
				a1, a2, a3 = self.components
				b1, b2, b3 = operand.components

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
		return self.__describe__()

	def distance(self, operand):
		"""Returns the euclidean distance between two Vectors.

		Parameters
		----------
		operand : a Vector

		"""

		if isinstance(operand, Vector):
			a, b = self.__dress__(operand)
			dist = sqrt(sum([pow((x - y), 2) for x, y in zip(a, b)]))

			return dist
		else:
			raise TypeError(
				"distance() requires <Vectors> types"
			)

	def dotmul(self, operand):
		"""Returns the dot product of two vectors.

		Parameters
		----------
		operand : a Vector

		"""

		if isinstance(operand, Vector):
			a, b = self.__dress__(operand)
			dmul = sum([x*y for x,y in zip(a, b)])

			return dmul
		else:
			raise ValueError(
				"dotmul() requires only <Vector> types"
			)

	def extend(self, desired_length, extension_component=0):
		"""Extends a Vector with several of a specified component.

		Parameters
		----------
		desired_length : The expected number of dimensions for the extended Vector

		extension_component : The number to use to extend the Vector. 
					 Defaults to 0.

		"""

		self.components = self.extended(
			desired_length, extension_component
		).components
		self.dimensions = self.__dimensions__()

	def extended(self, desired_length, extension_component=0):
		"""Returns a new Vector, which is the original Vector extended with several of a specified component.

		Example
		-------
		if x = Vector(2,3),
		x.extended(4, 1) returns Vector(2.0 3.0 1.0 1.0)

		*** x is preserved as a new Vector is returned.

		Parameters
		----------
		desired_length : The expected number of dimensions for the extended Vector

		extension_component : The number to use to extend the Vector. Defaults to 0.
		
		"""

		if (
			type(desired_length)==type(extension_component)==int and
			desired_length > 0
		):
			deficit = desired_length - len(self)
			extend_list = [extension_component for i in range(deficit)]
			ext_vector = Vector(self.components + extend_list)

			return ext_vector
		else:
			raise ValueError(
				"All arguments to extended() should be valid positive integers"
			)

	def insert(self, index, value):
		"""Inserts a new component.

		Parameters
		----------
		index : Index where new component must be inserted

		value : Component to be inserted

		"""

		if type(value) in (int, float):
			self.components.insert(index, value)
			self.dimensions = self.__dimensions__()
		else:
			raise TypeError(
				"value must be an integer or float"
			)

	def pop(self, index=None):
		"""Deletes a component.

		Parameters
		----------
		index : Index of component to be deleted.
				Defaults to the index of the last component.

		"""

		if  (type(index)==int and index >= 0) or index==None:
			index = index if index!=None else self.dimensions-1
			self.components.pop(index)

			if self.__dimensions__()==1:
				self.components.append(0)

			self.dimensions = self.__dimensions__()
		else:
			raise ValueError(
				"index argument must be a positive integer"
			)

	def reverse(self):
		"""Sets a Vector in the opposite direction."""

		old_self_coms = self.components[:]
		self.components = [num*-1 for num in self.components]

	def reversed(self):
		"""Returns a new Vector whose direction is opposite that of the original Vector.

		Unlike reverse(), this preserves original Vector.
		"""

		r_vec = self.smul(-1)		
		return r_vec

	def smul(self, scalar):
		"""Returns the product of a scalar multiplication.

		Parameters
		----------
		scalar : a number of int or float type to multiply vector.

		verbose : if true, a return string gives details on the calculation.

		"""

		if type(scalar) in (int, float):
			self_to_list = list(self.components)
			self_to_list = [
				scalar * num for num in self_to_list
			]

			result = Vector(self_to_list)
			return result
		else:
			raise ValueError(
				"Second operand must be a scalar"
			)

	def subtract(self, operand):
		"""Performs Vector subtraction.

		Parameters
		----------
		operand : a Vector

		"""

		return self.__sub__(operand)


	def subvec(self, start, end):
		"""Creates a new Vector from a sequence of components from anoperand Vector.

		Parameters
		----------
		start : the beggining index of the selected sequence

		end : the ending index of the selected sequence

		"""

		return Vector(self.components[start:end])

	def to_list(self):
		"""Returns a list of a Vector's components."""

		return self.components

	def to_str(self):
		"""Returns a string representation of a Vector."""

		return self.__repr__()

	def to_tuple(self):
		"""Returns a string representation of a Vector."""

		return tuple(self.components)


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


def main():

	interactive_shell_header = (
		"====================================================="
		"\n\n"
		"\tVECTORKIT :: v0.1.3 \n"
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