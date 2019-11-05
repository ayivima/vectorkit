VECTOR KIT
==========

Vectorkit is part of a series of implementations of mathematical concepts of AI from the scratch. 

.. image:: https://raw.githubusercontent.com/ayivima/vectorkit/master/img/shell_img.png


ATTRIBUTES AND METHOD OVERVIEW
==============================


Vector Properties
-----------------

``components`` - a list of the components of a vector

``dimensions`` - the dimension of the vector, or count of its components

``min`` - the minimum component

``max`` - the maximum component

``sum`` - the sum of the components of a vector

``memsize`` - the size of a vector in memory


Vector Methods
--------------

``add`` - Adds two vectors

``append`` - Appends new components to a vector

``concat`` - Merges two vectors into a single vector

``corr`` - Returns the correlation of two vectors

``cov`` - Returns the covariance between two vectors

``crossmul`` - Returs the cross product of two 3-dimensional vectors

``describe`` - Returns a description of a vector, including its dimensions and memory size

``distance`` - Returns the euclidean distance between two vectors

``dotmul`` - Returns the dot product between two vectors

``insert`` - Inserts a new component at a specified index

``magnitude`` - Returns the magnitude of a vector

``mean`` - Returns the mean of the components of a vector

``minmax`` - Returns a variant of a vector which has been normalized using standard min-max feature scaling

``minmaxmean`` - Returns a variant of a vector which has been normalized using standard mean and min-max feature scaling

``mse`` - Returns the mean square error of two vectors

``normalize`` - Returns a variant of a vector which has been normalized using the z-score or standard deviation

``pad`` - Appends zeroes to vectors to a specified length, in-place

``padded`` - Returns a new vector with zero appended to it to a specified length,

``pop`` - Removes a component at a specified location

``relu`` - Passes a vector through a Rectified Linear Unit function and returns a new vector

``reverse`` - Reverses the direction of a vector in-place

``reversed`` - Returns a variant of a vector with reversed direction

``sdiv`` - Returns a new vector, which is the quotient from a scalar division of a vector

``shuffle`` - Shuffles vector components in place

``shuffled`` - Returns a new vector with shuffled version of a vector's components

``sigmoid`` - Passes a vector through a logistic sigmoid function and returns a new vector

``softmax`` - Passes a vector through a softmax function and returns a new vector

``smul`` - Returns a new vector, which is the product from a scalar multiplication of a vector

``std`` - Returns the standard deviation of the components of a vector

``stdnorm`` - Returns a variant of a vector which has been normalized using the z-score or standard deviation

``subtract`` - Returns a new vector, which is the result of the subtraction of one vector from another

``subvec`` - Returns a new vector which is a slice from the original vector

``tanh`` - Passes a vector through a TanH function and returns a new vector

``to_list`` - Returns a list of the components of a vector

``to_tuple`` - Returns a tuple of the components of a vector

``unitvec`` - Returns a new vector which has been scaled to unit length

``vector_eq`` - Returns the vector equation of a line between two vectors


Others
------

``isovector`` -  Returns a vector of a specified length containing the same component throughout

``randvec`` - Generates a random vector of specified length


WORK IN PROGRESS
================

New Methods
-----------

``jaccard`` - Returns the jaccard similarity between two vectors

``leakyrelu`` - Passes vector through the leaky version of Rectified Linear Unit

``leastdev`` - Returns the Least Absolute Deviations(L1 Norm) between to vectors

``leastsq`` - Returns the Least Squares(L2 Norm) of two vectors

``mae`` -  Returns the mean absolute error between two vectors

``pararelu`` - Passes vector through the parametric version of Rectified Linear Unit

``rmse`` -   Returns the root mean square error between two vectors

``rsquare`` - Calculates the R square error between two vectors


PYPI VERSION HISTORY
====================
0.1.6
-----
Bug fix

0.1.5
-----
Added new methods: ``corr``, ``cov``, ``mse``, ``relu``, ``sigmoid``, ``softmax``, ``tanh``

Changing method names ``extend`` and ``extended`` to ``pad`` and ``padded`` respectively.

0.1.4
-----
Added functionality

0.1.3
-----
First Tested Version

0.1.0
-----
First Version with basic functionality


AUTHOR
======

Victor Mawusi Ayi <ayivima@hotmail.com>

