
"""Tests for Vectorkit"""

from math import sqrt, floor
import random
import sys
import unittest

from vectorkit import Vector, isovector, randvec


__name__ = "VectorkitTester"
__version__ = "0.2.0"
__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


# Some Test Vectors
x = Vector(1,2)
y = Vector(2,1)
z = Vector(1,2,3)


class VectorToolsTester(unittest.TestCase):

	def test_addition(self):
		x = Vector(1,2)
		y = Vector(2,1)

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
		c.append([4,0.5])

		self.assertEqual(Vector(1,2,3,4), a)
		self.assertEqual(Vector(1,2,3,4,5), b)
		self.assertEqual(Vector(1,2,3,4,0.5), c)

	def test_chaining_valid_operations(self):
		a = Vector(1,2,3)
		b = Vector(1,2,3)

		# chain addition and subtraction
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
		x = Vector(1,2)

		self.assertTrue(2 in x)
		self.assertFalse(4 in x)
		
	def test_corr(self):
		a = Vector(2, 3, 5)
		b = Vector(6, 5, 4)
		c = round(a.corr(b), 2)
		expected_c = -0.98

		self.assertEqual(c, expected_c)

	def test_cosine_similarity(self):
		#test orthogonal vectors
		a = Vector(1, 0)
		b = Vector(0, 1)
		expected_cosinesim = 0
		cosinesim = a.cosinesim(b)
		
		self.assertEqual(expected_cosinesim, cosinesim)
		
		#test diametrically opposed vectors
		a = Vector(2, 3)
		b = Vector(-2, -3)
		expected_cosinesim = -1
		cosinesim = int(a.cosinesim(b))
		
		self.assertEqual(expected_cosinesim, cosinesim)
		
		#test vectors of similar orientation
		a = Vector(2, 3)
		b = Vector(2, 3)
		expected_cosinesim = 1
		cosinesim = int(a.cosinesim(b))
		
		self.assertEqual(expected_cosinesim, cosinesim)

	def test_cost_function(self):
		a = Vector(1, 2, 3)
		b = Vector(0.5, 1, 1.5)
		
		cost = a.cost(b)
		expected_cost = 3.5/6
		
		self.assertEqual(expected_cost, cost)

	def test_cov(self):
		a = Vector(2, 3, 5)
		b = Vector(6, 5, 4)
		c = Vector(3, 4, 5)
		d = Vector(1, 1, 1)
		z = a.cov(b), a.cov(c), a.cov(d)
		expected_z = (-1.0, 1.0, 0.0)

		self.assertEqual(z, expected_z)

	def test_cross_product(self):
		a = Vector(3, -3, 1)
		b = Vector(4, 9, 2)
		expected_cross_product = Vector(-15, -2, 39)

		self.assertEqual(expected_cross_product, a.crossmul(b))

	def test_describe(self):
		x = Vector(1,2)

		expected_x_description = (
			"A 2-dimensional vector. "
			"[ Memory Size 190 bytes ]"
		)
		actual_x_description = x.describe()

		self.assertEqual(expected_x_description, actual_x_description)

	def test_dimensions(self):
		x = Vector(1,2)
		z = Vector(1,2,3)
	
		expected_x_dimensions = 2
		expected_z_dimensions = 3

		self.assertEqual(expected_x_dimensions, x.dimensions)
		self.assertEqual(expected_z_dimensions, z.dimensions)

	def test_distance(self):
		x = Vector(1,2)
		y = Vector(2,1)

		expected_distance_x_y = sqrt(2)
		self.assertEqual(expected_distance_x_y, x.distance(y))

	def test_dot_product(self):
		x = Vector(1,2)
		y = Vector(2,1)
		z = Vector(1,2,3)
		
		expected_dot_product_x_y = 4
		expected_dot_product_x_z = 5

		self.assertEqual(expected_dot_product_x_y, x.dotmul(y))
		self.assertEqual(expected_dot_product_x_z, x.dotmul(z))
	
	def test_element_wise_multiplication(self):
		x = Vector(1,2)
		y = Vector(2,1)
		
		expected_xy = Vector(2, 2)
		xy = x.emul(y)
		
		self.assertEqual(expected_xy, xy)

	def test_element_wise_division(self):
		x = Vector(1,2)
		y = Vector(2,1)
		
		expected_x_div_y = Vector(0.5, 2)
		x_div_y = x.ediv(y)
		
		self.assertEqual(expected_x_div_y, x_div_y)

	def test_equality(self):
		self.assertTrue(Vector(1,3,4)==Vector(1,3,4))
		self.assertTrue(Vector(1)==Vector(1,0))
		self.assertFalse(Vector(1,2,4)==Vector(1,3,4))
		self.assertFalse(Vector(1,-3,4)==Vector(1,3,4))

	def test_padding(self):
		# Test in-place extension using .extend() with default 
		# extension component.
		# Additional test that vector 'a' is changed by in-place extension
		a = Vector(1, 2)
		a.pad(4)
		expected_padded_a = Vector(1,2,0,0)

		self.assertEqual(expected_padded_a, a)
		self.assertFalse(Vector(1,2)==a)

		# Test in-place extension with custom extension component
		a = Vector(1, 2)
		a.pad(5, 3)
		expected_extended_a = Vector(1,2,3,3,3)

		# Test for non in-place extension using .extended() :)
		b = Vector(1,3)
		
		c = b.padded(4) # using default extension component
		d = b.padded(4, 10)

		expected_c = Vector(1,3,0,0)
		self.assertEqual(expected_c, c)

		expected_d = Vector(1,3,10,10)
		self.assertEqual(expected_d, d)

		# Test that vector 'b' is preserved
		self.assertTrue(Vector(1,3)==b)

	def test_flatten(self):
		x = [Vector(2,3), 5, (3,5), [3,5]]

		y = Vector.flatten(x)
		expected_y = Vector(2, 3, 5, 3, 5, 3, 5)
		
		self.assertEqual(expected_y, y)
	
	def test_jaccard(self):
		a = Vector(1, 2, 3, 4)
		b = Vector(1, 2, 4, 4)
		
		expected_jaccard = round(3/4, 2)
		jaccard = a.jaccard(b)
		
		self.assertEqual(expected_jaccard, jaccard)
	
	def test_join(self):
		a = Vector(1, 2, 3)
		
		expected_join = "1.0 2.0 3.0"
		actual_join = a.join()
		
		self.assertEqual(expected_join, actual_join)

	def test_least_absolute_deviations(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)
		
		expected_lad = 1
		lad = a.leastdev(b)
		
		self.assertEqual(expected_lad, lad)

	def test_least_squares(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)
		
		expected_lsq = 1
		lsq = a.leastsq(b)
		
		self.assertEqual(expected_lsq, lsq)

	def test_vector_concatenation(self):
		x = Vector(1,2)
		y = Vector(2,1)
		expected_result_vector = Vector(1,2,2,1)

		self.assertEqual(expected_result_vector, x.concat(y))

	def test_vectors_dressing(self):
		# Test vector 'dressing'
		a = Vector(1,1,1,1)
		b = Vector(1,2)
		expected_return_from_a_dress_b = ([1,1,1,1],[1,2,0,0])

		self.assertEqual(expected_return_from_a_dress_b, a._dress_(b))

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

	def test_magnitude(self):
		a = Vector(1,2,3)
		
		expected_magnitude = sqrt(14)
		self.assertEqual(expected_magnitude, a.magnitude())

	def test_mean(self):
		X = Vector(2, 7, 3)

		# Test mean
		expected_mean = 4
		self.assertEqual(expected_mean, X.mean())

	def test_memsize(self):
		a = Vector(1,2,3)

		tsize = 0
		props = (
			a,
			a.components,
			a.dimensions,
			a.min,
			a.max,
			a.sum,
		)
		for x in props:
			tsize += sys.getsizeof(x)

		self.assertEqual(tsize, a.memsize)

	def test_mae(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)
		
		expected_mae = 1/3
		mae = a.mae(b)
		
		self.assertEqual(expected_mae, mae)
		
	def test_mbe(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)
		
		expected_mbe = -1/3
		mbe = a.mbe(b)
		
		self.assertEqual(expected_mbe, mbe)
		
	def test_mse(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)
		
		expected_mse = 1/3
		mse = a.mse(b)
		
		self.assertEqual(expected_mse, mse)

	def test_multiplication_with_operator(self):
		# Using (*) between vectors produces dot product
		# Testing dressing in the background
		
		expected_dot_product_x_y = 4
		expected_dot_product_x_z = 5

		self.assertEqual(expected_dot_product_x_y, x*y)
		self.assertEqual(expected_dot_product_x_z, x*z)

	def test_normalization_techniques(self):
		X = Vector(1, 2, 3)
				
		# test regular minmax
		expected_return = Vector(0, 0.5, 1)
		self.assertEqual(expected_return, X.minmax())
		self.assertEqual(expected_return, X.normalize("minmax"))
		
		# test regular minmaxab
		expected_return = Vector(2, 2.5, 3)
		self.assertEqual(expected_return, X.minmax(2, 3))
		
		
		# test minmaxmean
		expected_return = Vector(-0.5, 0, 0.5)
		self.assertEqual(expected_return, X.minmaxmean())
		self.assertEqual(expected_return, X.normalize("minmaxmean"))

		# test standard/zscore normalisation
		expected_return = Vector(-1.224744871391589, 0.0, 1.224744871391589)
		self.assertEqual(expected_return, X.stdnorm())
		self.assertEqual(expected_return, X.normalize("zscore"))

		# test scaling to unit length
		X = Vector(4, -9)
		expected_return = Vector(4/sqrt(97), -9/sqrt(97))
		self.assertEqual(expected_return, X.unitvec())

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
	
	def test_power(self):
		a = Vector(1,2,3)

		expected_power = Vector(1, 8, 27)
		power = a.pow(3)

		self.assertEqual(expected_power, power)

	def test_randvec(self):
		seed = 2
		dimensions = 2
		random.seed(seed)

		a = randvec(dimensions, seed)
		expected_a = Vector(
			[random.random()*2-1 for i in range(dimensions)]
		)

		self.assertEqual(2, a.dimensions)
		self.assertTrue(isinstance(a, Vector))

	def test_random_vector(self):
		seed = 3
		dimensions = 2
		random.seed(seed)

		a = Vector.random(dimensions, seed)
		expected_a = Vector(
			[random.random()*2-1 for i in range(dimensions)]
		)

		self.assertEqual(2, a.dimensions)
		self.assertTrue(isinstance(a, Vector))

	def test_rmse(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)

		expected_rmse = sqrt(1/3)
		rmse = a.rmse(b)

		self.assertEqual(expected_rmse, rmse)

	def test_rsquare(self):
		a = Vector(1, 2, 3)
		b = Vector(2, 2, 3)

		expected_r2 = 1 - ((1/3)/(2/3))
		r2 = a.rsquare(b)

		self.assertEqual(expected_r2, r2)
		
	def test_relu(self):
		# Test positive and negative values
		a = Vector(-1,-2,-3, 1)
		b = a.relu()
		expected_relu = Vector(0, 0, 0, 1)

		self.assertEqual(b, expected_relu)

		# Test positive values
		a = Vector(1, 2, 3, 1)
		b = a.relu()
		expected_relu = Vector(1, 2, 3, 1)

		self.assertEqual(b, expected_relu)
		
	def test_relu_leaky(self):
		# Test positive and negative values
		a = Vector(-1,-2,-3, 1)
		b = a.leakyrelu()
		expected_relu = Vector(-0.01, -0.02, -0.03, 1)

		self.assertEqual(b, expected_relu)
		
		# Test positive values
		a = Vector(1, 2, 3, 1)
		b = a.leakyrelu()
		expected_relu = Vector(1, 2, 3, 1)
		
		self.assertEqual(b, expected_relu)

	def test_relu_parametric(self):
		# Test positive and negative values
		a = Vector(-1,-2,-3, 1)
		b = a.pararelu(0.5)
		expected_relu = Vector(-0.5, -1, -1.5, 1)

		self.assertEqual(b, expected_relu)
		
		# Test positive values
		a = Vector(1, 2, 3, 1)
		b = a.pararelu(0.5)
		expected_relu = Vector(1, 2, 3, 1)
		
		self.assertEqual(b, expected_relu)	

	def test_scalar_multiplication(self):
		a = Vector(1,2,3)
		expected_result_from_scalar_mul_by_two = Vector(2,4,6)

		# Test .smul()
		self.assertEqual(expected_result_from_scalar_mul_by_two, a.smul(2))

		# Test scalar multiplication with (*) operator
		# Test multiplication from right and left sides
		self.assertEqual(expected_result_from_scalar_mul_by_two, 2*a)
		self.assertEqual(expected_result_from_scalar_mul_by_two, a*2)

	def test_shuffling(self):
		# Test shuffled()
		a = Vector(1,2,3,4,5,6,7,8,9,10,11,12)
		b = a.shuffled()

		self.assertEqual(a.sum, b.sum)
		self.assertEqual(a.mean(), b.mean())
		self.assertEqual(a.std(), b.std())

		# Test shuffled
		a = Vector(1,2,3,4,5,6,7,8,9,10,11,12)
		b = Vector(1,2,3,4,5,6,7,8,9,10,11,12)
		
		a.shuffle()
		self.assertTrue(a != b)
		
	def test_sigmoid(self):
		
		a = Vector(1,2,3)
		b = a.sigmoid()
		expected_sig = Vector(
			0.7310585786300049,
			0.8807970779778823,
			0.9525741268224334
		)
		
		self.assertEqual(b, expected_sig)
		
	def test_softmax(self):
		
		a = Vector(1,2,3)
		b = a.softmax()
		expected_soft = Vector(
			0.09003057317038046,
			0.24472847105479767,
			0.6652409557748219
		)
		
		self.assertEqual(b, expected_soft)

	def test_std(self):
		a = Vector(1,2,3,4,5)
		
		# check standard devaition
		expected_result = sqrt(2)
		self.assertEqual(expected_result, a.std())

	def test_string_representation(self):
		a = Vector(1,2,3)

		self.assertEqual("Vector(1.0 2.0 3.0)", str(a))
		self.assertEqual(
			"The right representation must be: Vector(1.0 2.0 3.0)", 
			"The right representation must be: {}".format(a)
		)
		
		self.assertEqual("1.0 2.0 3.0", a.__repr__(matrixmode=True))

	def test_subtraction(self):
		expected_x_minus_y = Vector(-1,1)
		expected_y_minus_x = Vector(1,-1)

		# Test subtraction using (-) operator
		self.assertEqual(expected_x_minus_y, (x - y))
		self.assertEqual(expected_y_minus_x, (y - x))

		# Test subtraction using .subtract() method
		self.assertEqual(expected_x_minus_y, x.subtract(y))
		self.assertEqual(expected_y_minus_x, y.subtract(x))

	def test_sum_min_max(self):
		a = Vector(1,2,3,4,5)
		
		# after initialisation
		expected_min = (1.0, 0)
		expected_max = (5.0, 4)
		expected_sum = 15

		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)

		# after changing a component
		a[1] = -2
		expected_min = (-2.0, 1)
		expected_max = (5.0, 4)
		expected_sum = 11

		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)

		# after deletion
		a.pop(1)
		expected_min = (1.0, 0)
		expected_max = (5.0, 3)
		expected_sum = 13

		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)
		
		# after appending a scalar
		a.append(-2)
		expected_min = (-2.0, 4)
		expected_max = (5.0, 3)
		expected_sum = 11

		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)

		# after inserting a value
		a.insert(2, -20)
		expected_min = (-20.0, 2)
		expected_max = (5.0, 4)
		expected_sum = -9
		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)
		
		# after inserting a value at min position
		a.insert(2, -21)
		expected_min = (-21.0, 2)
		expected_max = (5.0, 5)
		expected_sum = -30
		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)
		
		# after inserting a value at max position
		a.insert(5, 10)
		expected_min = (-21.0, 2)
		expected_max = (10, 5)
		expected_sum = -20
		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)
		
		# after appending a sequence
		a.append([-22, 20, 1])
		expected_min = (-22.0, 8)
		expected_max = (20.0, 9)
		expected_sum = -21.0

		self.assertEqual(expected_min, a.min)
		self.assertEqual(expected_max, a.max)
		self.assertEqual(expected_sum, a.sum)

	def test_sum_of_vectors(self):
		a = Vector(1,1,1)
		b = Vector(2,2,2)
		c = Vector(3,3,3)
		
		expected_sum = Vector(6,6,6)
		sum = Vector.sum([a, b, c])
		
		self.assertEqual(expected_sum, sum)
	
	def test_tanh(self):
		x = Vector(1,2,3)
		tanh = x.tanh()
		expected_tanh = Vector(
			0.7615941559557646,
			0.9640275800758169,
			0.9950547536867307
		)
		
		self.assertEqual(expected_tanh, tanh)
	
	def test_vector_equation(self):
		a = Vector(1, 2, 3)
		b = Vector(3, 2, 1)
		
		expected = "eq = [1.0 2.0 3.0] + t[2.0 0.0 -2.0]"
		
		self.assertEqual(expected, a.vector_eq(b))

	def test_vectorization_of_dictionaries(self):
		a = {"latitude":2.345, "longitude":-3.421}

		expected_vector_from_a = Vector(2.345, -3.421)
		self.assertEqual(expected_vector_from_a, Vector(a))
	
	def test_vectorization_of_lists(self):
		a = [1,2,3]

		expected_vector_from_a = Vector(1,2,3)
		self.assertEqual(expected_vector_from_a, Vector(a))

	def test_vectorization_of_range(self):
		a = range(1, 4)

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
	
