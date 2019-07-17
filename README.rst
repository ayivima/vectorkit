VECTOR KIT
==========

Vectorkit, inspired by 60DAYSOFUDACITY, is part of a series of implementations of mathematical concepts of AI from the scratch. 
Vectorkit seeks to make vector arithmetic simple and convenient for everyone. It may serve as a utility in a 
large ecosystem of scientific libraries or, more simply, as a toy to be played with to understand Vector math.


INSTALLATION
============

::

    $ pip install vectorkit


VECTORKIT INTERACTIVE SHELL
===========================
You can use the interactive shell for live vector arithmetic. To activate the interactive shell, type `vectorkit` as demonstrated below.

::

    $ vectorkit
    =====================================================

        VECTORKIT :: v0.1.3
        Interactive Shell

    =====================================================
    >>>>
    >>>> Vector(2,3,4)
    Vector(2.0 3.0 4.0)



REGULAR USAGE
=============

To use vectorkit in your script, import it the usual way.

::

    >>> from vectorkit import Vector, isovector
    
::

    >>> import vectorkit
    

Create a vector from given numbers
----------------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w
    Vector(1.0 2.0 3.0)


Get vector components
---------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w.components
    [1, 2, 3]
    

Get Vector's number of dimensions
---------------------------------

:Using the .dimensions:

::

    >>> w = Vector(1, 2, 3)
    >>> w.dimensions
    3
    
:Using len():

::

    >>> w = Vector(1, 2, 3)
    >>> len(w)
    3
    
    
Describe a Vector
-----------------

::

    >>> w = Vector(1, 2, 3)
    >>> w.describe()
    A 3-dimensional vector with components: 1, 2, 3


Check component membership
--------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w
    Vector(1 2 3)
    >>>
    >>> 2 in w
    True
    >>>
    >>> 5 in w
    False


Compare two Vectors
-------------------

:Equality:

::

    >>> w = Vector(1, 2, 3)
    >>> x = Vector(1, 2, 3)
    >>> y = Vector(2, 4, 6)
    >>> w==x
    True
    >>>
    >>> x==y
    False
 
:Inequality:
 
::

    >>> w = Vector(1, 2, 3)
    >>> x = Vector(1, 2, 3)
    >>> y = Vector(2, 4, 6)
    >>> w!=x
    False
    >>>
    >>> x!=y
    True


Vector Addition
---------------

First Option

::

    >>> w = Vector(1, 2, 3)
    >>> x = Vector(3, 2, 1)
    >>> w + x
    Vector(4.0 4.0 4.0)
   
::

    >>> w = Vector(1, 2, 3, 4, 5)
    >>> x = Vector(3, 2, 1)
    >>> w + x
    Vector(4.0 4.0 4.0 4.0 5.0)


Second Option

::

    >>> w = Vector(1, 2, 3)
    >>> x = Vector(3, 2, 1)
    >>> vector_sum = w.add(x)
    >>> vector_sum
    Vector(4.0 4.0 4.0)
    

Vector Subtraction
------------------

First Option

::

    >>> a = Vector(2, 2, 2)
    >>> b = Vector(1, 1, 1)
    >>> a - b
    Vector(1.0 1.0 1.0)

Second Option

::

    >>> a = Vector(2, 2, 2)
    >>> b = Vector(1, 1, 1)
    >>> vector_sub = a.subtract(b)
    >>> vector_sub
    Vector(1.0 1.0 1.0)
 
::

    >>> a = Vector(2, 2, 2, 2, 2)
    >>> b = Vector(1, 1, 1)
    >>> vector_sub = a.subtract(b)
    >>> vector_sub
    Vector(1.0 1.0 1.0 2.0 2.0)
    

Get an element or elements from a Vector
-----------------------------------------------------
Unlike .subvec(), standard indexing/slicing returns an int or float, if one element is requested or, a standard list of several items requested.


:Using standard slicing notation:

::

    >>> x = Vector(1, 2, 3, 4)
    >>> x[1:3]
    [2, 3]
    

Create a new Vector using a slice from another Vector
-----------------------------------------------------

:Using subvec():

::

    >>> x = Vector(1, 2, 3, 4)
    >>> new_vector = x.subvec(1, 3)
    >>> new_vector
    Vector(2.0 3.0)


Add new components to Vectors
-----------------------------

:Using append():

Append one value

::

    >>> w = Vector(1, 2, 3)
    >>> w.append(4)
    >>> w
    Vector(1.0 2.0 3.0 4.0)


Append several values bundled in a tuple or list

::

    >>> w = Vector(1, 2, 3)
    >>> w.append([4, 5, 6])
    >>> w
    Vector(1.0 2.0 3.0 4.0 5.0 6.0)

:Using insert(index, value):

::

    >>> w = Vector(1, 2, 3)
    >>> w.insert(2, 67)
    >>> w
    Vector(1.0 2.0 67.0 3.0)


Change a component's value
--------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w
    Vector(1.0 2.0 3.0)
    >>>
    >>> w[2] = 78
    >>> w
    Vector(1.0 2.0 78.0)


Delete a component
------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w.pop(1)
    Vector(1.0 3.0)
    


Extend a Vector by adding component a specified number of times
---------------------------------------------------------------

:Using extended(desired_length, extension_component):
``extended()`` returns a new extended Vector, and preserves the original vector.

::

    >>> w = Vector(1, 2, 3)
    >>> extended_vector = w.extended(6, 1)
    >>> extended_vector
    Vector(1.0 2.0 3.0 1.0 1.0 1.0)
    >>>
    >>> w
    Vector(1.0 2.0 3.0)   
    
    
:Using extended(desired_length):
Calling ``extended()`` without a ``extension_component`` uses 0 as fill value.

::

    >>> w = Vector(1, 2, 3)
    >>> extended_vector = w.extended(6)
    >>> extended_vector
    Vector(1.0 2.0 3.0 0.0 0.0 0.0)


:Using extend(desired_length, extension_component) or extend(desired_length):
``extend()`` does not preserve the original Vector; it changes it. 

::

    >>> w = Vector(1, 2, 3)
    >>> extended_vector = w.extend(6, 1)
    >>> extended_vector
    None
    >>>
    >>> w
    Vector(1.0 2.0 3.0 1.0 1.0 1.0)
    
Change a Vector's direction
---------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w
    Vector(1.0 2.0 3.0)
    >>>
    >>> w.reverse()
    None
    >>> w
    Vector(-1.0 -2.0 -3.0)
       
::

    >>> w = Vector(1, -2, 3)
    >>> w
    Vector(1.0 -2.0 3.0)
    >>>
    >>> w.reverse()
    None
    >>> w
    Vector(-1.0 2.0 -3.0)


Create a Vector that has an opposite direction to the current Vector
--------------------------------------------------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> w
    Vector(1.0 2.0 3.0)
    >>>
    >>> new_vector = w.reversed()
    >>> new_vector
    Vector(-1.0 -2.0 -3.0)



Scalar Multiplication
---------------------

:Using smul():

::

    >>> x = Vector(3, 2, 1)
    >>> w.smul(3)
    Vector(9.0 6.0 3.0)



Dot product of two Vectors
--------------------------

:Use dotmul():

::

    >>> w = Vector(1, 2, 3, 4, 5)
    >>> x = Vector(3, 2, 1)
    >>> w.dotmul(x)
    10
    

Cross Product of two vectors within 3-D space
---------------------------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> x = Vector(3, 2, 1)
    >>> w.crossmul(x)
    Vector(-4.0 8.0 -4.0)


Distance between two vectors
----------------------------

::

    >>> w = Vector(1, 2, 3)
    >>> x = Vector(3, 2, 1)
    >>> w.distance(x)
    2.8284271247461903


Create a homogenous Vector of a specified dimension
---------------------------------------------------

::

    >>> w = isovector(2, 4)
    >>> w
    Vector(2.0 2.0 2.0 2.0)


Transforms a valid sequence or single numerical value(int or float) into a Vector
---------------------------------------------------------------------------------

::

    >>> w = Vector([2, 4])
    >>> w
    Vector(2.0 4.0)
    >>>
    >>> Vector((1, 9))
    Vector(1.0 9.0)
    

VERSION HISTORY
===============

0.1.3
-----
First Version with extensive tests

0.1.0
-----
First Version with basic functionality


AUTHOR
======

Victor Mawusi Ayi

