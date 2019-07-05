
"""Tests for Vectorkit."""

from math import sqrt
import unittest

from vectorkit import Vector, isovector


__name__ = "VectorkitTester"
__version__ = "0.1.3"
__author__ = "Victor Mawusi Ayi"


# Test Vectors
x = Vector(1,2)
y = Vector(2,1)
z = Vector(1,2,3)


class VectorToolsTester(unittest.TestCase):

	def test_addition(self):
		expected_sum = Vector(3,3)

		# Test addition using (+) operator
		self.assertEqual(expected_sum, (x + y))
		self.assertEqual(expected_sum, (y + x))

		# Test addition using .add() method
		self.assertEqual(expected_sum, x.add(y))
		self.assertEqual(expected_sum, y.add(x))

	def test_append(self):
		a = Vector(1,2,3)
		b = Vector(1,2,3)
		c = Vector(1,2,3)

		a.append(4)
		b.append((4,5))
		c.append([4,5])

		self.assertEqual(Vector(1,2,3,4), a)
		self.assertEqual(Vector(1,2,3,4,5), b)
		self.assertEqual(Vector(1,2,3,4,5), c)

	def test_chaining_valid_operations(self):
		a = Vector(1,2,3)
		b = Vector(1,2,3)

		chain1 = a + b - a + b
		chain1_expected_result = Vector(2,4,6)
		self.assertEqual(chain1_expected_result, chain1)

		# chain scalar multiplication and dot product
		chain2 = a * b * 5 * a
		chain2_expected_result = Vector(70,140,210)
		self.assertEqual(chain2_expected_result, chain2)

		# chain dot product, scalar multiplication, reversal
		chain3 = (a * b * 5 * a).smul(0.5).reversed()
		chain3_expected_result = Vector(-35,-70,-105)
		self.assertEqual(chain3_expected_result, chain3)

		# chain cross product, vector addition, 
		# vector subtraction, scalar multiplication
		a = Vector(3, -3, 1)
		b = Vector(4, 9, 2)
		chain4 = a.crossmul(b).smul(-1).subtract(b).add(a)
		chain4_expected_result = Vector(14,-10,-40)
		self.assertEqual(chain4_expected_result, chain4)

	def test_component_membership(self):
		self.assertTrue(2 in x)
		self.assertFalse(4 in x)

	def test_cross_product(self):
		a = Vector(3, -3, 1)
		b = Vector(4, 9, 2)
		expected_cross_product = Vector(-15, -2, 39)

		self.assertEqual(expected_cross_product, a.crossmul(b))

	def test_describe(self):
		expected_x_description = (
			"A 2-dimensional vector with components: 1.0, 2.0"
		)
		actual_x_description = x.describe()

		self.assertEqual(expected_x_description, actual_x_description)

	def test_dimensions(self):
		expected_x_dimensions = 2
		expected_z_dimensions = 3

		self.assertEqual(expected_x_dimensions, x.dimensions)
		self.assertEqual(expected_z_dimensions, z.dimensions)

	def test_distance(self):
		expected_distance_x_y = sqrt(2)
		self.assertEqual(expected_distance_x_y, x.distance(y))

	def test_dot_product(self):
		expected_dot_product_x_y = 4
		expected_dot_product_x_z = 5

		self.assertEqual(expected_dot_product_x_y, x.dotmul(y))
		self.assertEqual(expected_dot_product_x_z, x.dotmul(z))

	def test_equality(self):
		self.assertTrue(Vector(1,3,4)==Vector(1,3,4))
		self.assertTrue(Vector(1)==Vector(1,0))
		self.assertFalse(Vector(1,2,4)==Vector(1,3,4))
		self.assertFalse(Vector(1,-3,4)==Vector(1,3,4))

	def test_extension(self):
		# Test in-place extension using .extend() with default 
		# extension component.
		# Additional test that vector 'a' is changed by in-place extension
		a = Vector(1, 2)
		a.extend(4)
		expected_extended_a = Vector(1,2,0,0)

		self.assertEqual(expected_extended_a, a) 
		self.assertFalse(Vector(1,2)==a)

		# Test in-place extension with custom extension component
		a = Vector(1, 2)
		a.extend(5, 3)
		expected_extended_a = Vector(1,2,3,3,3)

		# Test for non in-place extension using .extended() :)
		b = Vector(1,3)
		c = b.extended(4) # using default extension component
		d = b.extended(4, 10)
		expected_c = Vector(1,3,0,0)
		expected_d = Vector(1,3,10,10)

		self.assertEqual(expected_c, c)
		self.assertEqual(expected_d, d)

		# Test that vector 'b' is preserved
		self.assertTrue(Vector(1,3)==b)

	def test_vector_concatenation(self):
		# x = Vector(1,2)
		# y = Vector(2,1)
		expected_result_vector = Vector(1,2,2,1)

		self.assertEqual(expected_result_vector, x.concat(y))

	def test_vectors_dressing(self):
		# Test vector 'dressing'
		a = Vector(1,1,1,1)
		b = Vector(1,2)
		expected_return_from_a_dress_b = ([1,1,1,1],[1,2,0,0])

		self.assertEqual(expected_return_from_a_dress_b, a.__dress__(b))

	def test_vector_reversal(self):
		# Reversal sets a Vector in opposite direction.
		# Test in-place reversal
		a = Vector(1,-3,1)
		a.reverse()
		expected_reversed_a = Vector(-1,3,-1)

		self.assertEqual(expected_reversed_a, a)

		# Test non in-place reversal :)
		b = Vector(1,-3,1)
		c = b.reversed()
		expected_reverse_of_b = Vector(-1,3,-1)

		self.assertEqual(expected_reverse_of_b, c)
		self.assertEqual(Vector(1,-3,1), b) # checks that b is unchanged

	def test_insert(self):
		a = Vector(1,2,3)
		a.insert(1,2)
		expected_a_after_insert = Vector(1,2,2,3)
		
		self.assertEqual(expected_a_after_insert, a)

	def test_isovector(self):
		a = isovector(2, 4)
		expected_a = Vector(2,2,2,2)

		self.assertEqual(expected_a, a)

	def test_multiplication_with_operator(self):
		# Using (*) between vectors produces dot product
		# Testing dressing in the background
		
		expected_dot_product_x_y = 4
		expected_dot_product_x_z = 5

		self.assertEqual(expected_dot_product_x_y, x*y)
		self.assertEqual(expected_dot_product_x_z, x*z)

	def test_pop(self):
		a = Vector(1,2,3)
		a.pop()
		expected_a_after_pop = Vector(1,2)

		self.assertEqual(expected_a_after_pop, a)

		# Test pop() with index specified
		b = Vector(1,2,3)
		b.pop(1)
		expected_b_after_pop = Vector(1,3)

		self.assertEqual(expected_b_after_pop, b)

	def test_scalar_multiplication(self):
		a = Vector(1,2,3)
		expected_result_from_scalar_mul_by_two = Vector(2,4,6)

		# Test .smul()
		self.assertEqual(expected_result_from_scalar_mul_by_two, a.smul(2))

		# Test scalar multiplication with (*) operator
		# Test multiplication from right and left sides
		self.assertEqual(expected_result_from_scalar_mul_by_two, 2*a)
		self.assertEqual(expected_result_from_scalar_mul_by_two, a*2)

	def test_string_representation(self):
		a = Vector(1,2,3)

		"Vector({})"
		self.assertEqual("Vector(1.0 2.0 3.0)", str(a))
		self.assertEqual(
			"The right representation must be: Vector(1.0 2.0 3.0)", 
			"The right representation must be: {}".format(a)
		)

	def test_subtraction(self):
		expected_x_minus_y = Vector(-1,1)
		expected_y_minus_x = Vector(1,-1)

		# Test subtraction using (-) operator
		self.assertEqual(expected_x_minus_y, (x - y))
		self.assertEqual(expected_y_minus_x, (y - x))

		# Test subtraction using .subtract() method
		self.assertEqual(expected_x_minus_y, x.subtract(y))
		self.assertEqual(expected_y_minus_x, y.subtract(x))

	def test_vectorization_of_dictionaries(self):
		a = {"latitude":2.345, "longitude":-3.421}
		expected_vector_from_a = Vector(2.345, -3.421)

		self.assertEqual(expected_vector_from_a, Vector(a))
	
	def test_vectorization_of_lists(self):
		a = [1,2,3]
		expected_vector_from_a = Vector(1,2,3)

		self.assertEqual(expected_vector_from_a, Vector(a))

	def test_vectorization_of_scalars(self):
		expected_vector_from_vectorisation_of_2 = Vector(2, 0)

		self.assertEqual(expected_vector_from_vectorisation_of_2, Vector(2))

	def test_vectorization_of_tuples(self):
		a = (1,2,3)
		expected_vector_from_a = Vector(1,2,3)

		self.assertEqual(expected_vector_from_a, Vector(a))


if __name__ == "VectorkitTester":
	unittest.main(verbosity=2)