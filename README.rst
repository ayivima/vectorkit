VECTOR KIT
==========

Vectorkit, inspired by Facebook's Secure and Private AI Course, is part of a series of implementations of mathematical concepts of AI from the scratch. 
Vectorkit seeks to make vector arithmetic simple and convenient for everyone. It may serve as a utility in a 
large ecosystem of scientific libraries or, more simply, as a toy to be played with to understand Vector math.


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

``crossmul`` - Returs the cross product of two 3-dimensional vectors

``describe`` - Returns a description of a vector, including its dimensions and memory size

``distance`` - Returns the euclidean distance between two vectors

``dotmul`` - Returns the dot product between two vectors

``extend`` - Appends zeroes to vectors to a specified length, in-place

``extended`` - Returns a new vector with zero appended to it to a specified length,

``insert`` - Inserts a new component at a specified index

``magnitude`` - Returns the magnitude of a vector

``mean`` - Returns the mean of the components of a vector

``minmax`` - Returns a variant of a vector which has been normalized using standard min-max feature scaling

``minmaxmean`` - Returns a variant of a vector which has been normalized using standard mean and min-max feature scaling

``normalize`` - Returns a variant of a vector which has been normalized using the z-score or standard deviation

``pop`` - Removes a component at a specified location

``reverse`` - Reverses the direction of a vector in-place

``reversed`` - Returns a variant of a vector with reversed direction

``sdiv`` - Returns a new vector, which is the quotient from a scalar division of a vector

``shuffle`` - Shuffles vector components in place

``shuffled`` - Returns a new vector with shuffled version of a vector's components

``smul`` - Returns a new vector, which is the product from a scalar multiplication of a vector

``std`` - Returns the standard deviation of the components of a vector

``stdnorm`` - Returns a variant of a vector which has been normalized using the z-score or standard deviation

``subtract`` - Returns a new vector, which is the result of the subtraction of one vector from another

``subvec`` - Returns a new vector which is a slice from the original vector

``to_list`` - Returns a list of the components of a vector

``to_tuple`` - Returns a tuple of the components of a vector

``unitvec`` - Returns a new vector which has been scaled to unit length

``vector_eq`` - Returns the vector equation of a line between two vectors


Others
------

``isovector`` -  Returns a vector of a specified length containing the same component throughout

``randvec`` - Generates a vector of specified length having random components


Newly Added Methods [yet to be rolled out to pypi]
-----------

``sigmoid`` - Passes a vector through a logistic sigmoid function and returns a new vector

``softmax`` - Passes a vector through a softmax function and returns a new vector

``relu`` - Passes a vector through a Rectified Linear Unit function and returns a new vector

``tanh`` - Passes a vector through a TanH function and returns a new vector

``mse`` - Returns the mean square error of two vectors

``corr`` - Returns the correlation of two vectors

``cov`` - Returns 



PYPI VERSION HISTORY
====================

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

