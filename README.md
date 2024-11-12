# Python_Portfolio_ChristianRoss
This is all the coding that I did in 450C

### ANALYZING PATIENT DATA 1-3
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```
    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])

data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```

print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]

print(type(data))
```

    <class 'numpy.ndarray'>

print(data.shape)
```

    (60, 40)

print('firt value in data:', data[0,0])
```

    firt value in data: 0.0

print('middle value in data:', data[29, 19])
```

    middle value in data: 16.0

print(data[0:4, 0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]


small = data[:3, 36:]
```

print('small is:')
```

    small is:


print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]

# lets use a numpy function
print(numpy.mean(data))
```

    6.14875

maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)

print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566

maxval = numpy.amax(data)
minval = numpy.amin(data)
stdval = numpy.std(data)
```

print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566
print('maximum inflammation:', maxval)
print('minimum inflamation:', minval)
print('standard deviation:', stdval)
```

    maximum inflammation: 20.0
    minimum inflamation: 0.0
    standard deviation: 4.613833197118566

# Sometimes we want to look at variations in statistical values, such as maximum inflammation per patient, 
# or avergae from day one.

patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)

print('maximum inflammation for patient 0:', numpy.amax(patient_0))
```

    maximum inflammation for patient 0: 18.0

print('maximum inflammation for patient 2:', numpy.amax(data[2, :]))
```

    maximum inflammation for patient 2: 19.0

print(numpy.mean(data, axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]

print(numpy.mean(data, axis = 0).shape)
```

    (40,)

print(numpy.mean(data, axis = 1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]


### PYTHON FUNDAMENTALS

# Any python interpreter can be used as a calculator:
3 + 5 * 4
```
    23

# Lets save a value to a variable
weight_kg = 60
```

print(weight_kg)
```
    60

# weight0 = valid
# 0weight = invalid
# weight and Weight are different
```

# Types of data
# There are three common types of data
# Integer numbers
# floating point numbers
# Strings
```

# Floating point number
weight_kg = 60.3
```

# String comprised of Letters
patient_name = "Jon Smith"
```

patient_id = '001'
```

# Use variables in python

weight_lb = 2.2 * weight_kg

print(weight_lb)
```

    132.66


# Lets add a prefix to our patient id

patient_id = 'inflam_' + patient_id
print(patient_id)
```

    inflam_001

# Lets combine print statements

print(patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3

# we can call a function inside another function

print(type(60.3))

print(type(patient_id))
```

    <class 'float'>
    <class 'str'>

# We can also do calculations inside the print function

print('weight in lbs:', 2.2 * weight_kg)
```

    weight in lbs: 132.66

print(weight_kg)
```
    60.3
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0


### STORING VALUES IN LISTS

odds = [1, 3, 5, 7, 11]
print('odds are:', odds)
```

    odds are: [1, 3, 5, 7, 11]

print('first element:', odds [0])
print('last element:', odds[3])
print('"-1" element:', odds[-1])
```

    first element: 1
    last element: 7
    "-1" element: 11

names = ['Curie', 'Darwing', 'Turing'] # Typo in Darwin's name

print('names is originally:', names)

names[1] = 'Darwin' # Correct the name

print('final value of names:', names)
```
    names is originally: ['Curie', 'Darwing', 'Turing']
    final value of names: ['Curie', 'Darwin', 'Turing']

#name = 'Darwin'
#name[0] = 'd'
```
odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11, 11]
removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element: [3, 5, 7, 11, 11]
    removed_element: 1

odds.reverse()
print('odds after reversing:', odds)
```

    odds after reversing: [11, 11, 7, 5, 3]

odds = [3, 5, 7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]

odds = [3,5,7]
primes = list(odds)
primes. append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]

binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes[2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last:', last)
```

    group: Drosophila
    species: melanogaster
    autosomes: ['2', '3', '4']
    last: 4

date = 'Monday 4 January 2023'
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning index:', day)
```

    Using 0 to begin range: Monday
    Omitting beginning index: Monday


months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position:', sond)

sond = months[8:len(months)]
print('Using len() to get last entry:', sond)

sond = months[8:]
print('Ommiting ending index:', sond)
```

    With known last position: ['sep', 'oct', 'nov', 'dec']
    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
    Ommiting ending index: ['sep', 'oct', 'nov', 'dec']

### USING MULTIPLE FILES

odds = [1,3,5,7]
```

print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5
    7

odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5


odds = [1,3,5,7,9,11,13,15,17,19]

for num in odds:
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19


length = 0
names = ['Curie', 'Darwin', 'Turing']
for value in names:
    length = length + 1
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.

name = "Rosalind"
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing


print(len([0,1,2,3]))
```

    4


name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3


### MAKING CHOICES

num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done


num = 53
print('before conditional...')
if num > 100:
    print(num, 'is greater tha 100')
print('...after conditional')
```

    before conditional...
    ...after conditional

num = 14

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    14 is positive

if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else:
    print('at least one part if false')
```

    at least one part if false

if (-1 > 0) and (-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    both of these are false


# FUCNTIONS 1-4

fahrenheit_val = 99
celsius_val = ((fahrenheit_val - 32) *(5/9))

print(celsius_val)
```

    37.22222222222222

fahrenheit_val2 = 43
celsius_val2 = ((fahrenheit_val2 -32) *(5/9))

print(celsius_val2)
```

    6.111111111111112

def explicit_fahr_to_celsius(temp):
    # Assign the converted value to a variable
    converted = ((temp - 32)*(5/9))
    # Return the values of the new variable
    return converted
```

def fahr_to_celsius(temp):
    # Return inverted values more efficiently using the return function without creating
    # a new variable. This code does the same thing as the previous function but it is more
    # explicit in explaining how the return command works
    return ((temp - 32) * (5/9))
```

fahr_to_celsius(32)
```

    0.0


explicit_fahr_to_celsius(32)
```


    0.0


print('Freezing point of water:', fahr_to_celsius(32), 'C')
print('Boiling point of water:', fahr_to_celsius(212), 'C')
```

    Freezing point of water: 0.0 C
    Boiling point of water: 100.0 C


def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

print('freezing point of water in Kelvin:', celsius_to_kelvin(0.))
```

    freezing point of water in Kelvin: 273.15


def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print('boiling point of water in Kelvin:', fahr_to_kelvin(212.0))
```

    boiling point of water in Kelvin: 373.15


print('Again, temperature in Kelvin was:', temp_k)
```


temp_kelvin = fahr_to_kelvin(212.0)
print('Temperature in kelvin was:', temp_kelvin)
```

    Temperature in kelvin was: 373.15



temp_kelvin
```




    373.15



def print_temperatures():
    print('Temperature in Fahrenheit was :', temp_fahr)
    print('Temperature in Kelvin was:', temp_kelvin)
    
temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```

    Temperature in Fahrenheit was : 212.0
    Temperature in Kelvin was: 373.15

import numpy
import matplotlib
import glob
import matplotlib.pyplot
```

def visualize(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```

def detect_problems(filename):
    
    data = numpy.loadtxt(filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis=0)[20] == 20:
        print("Suspicious looking maxima!")
    elif numpy.sum(numpy.amin(data, axis=0)) == 0:
        print('Minima add up to zero!')
    else:
        print('Seems ok!')
```

filenames = sorted(glob.glob('inflammation*.csv'))

for filename in filenames[:3]:
    visualize(filename)
    detect_problems(filename)
```


![png](output_3_0.png)


    Suspicious looking maxima!



![png](output_3_2.png)


    Suspicious looking maxima!



![png](output_3_4.png)


    Minima add up to zero!


def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```

z = numpy.zeros((2,2))
print(offset_mean(z, 3))
```

    [[3. 3.]
     [3. 3.]]


data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

print(offset_mean(data, 0))
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]


print('original min, mean and max are:', numpy.amin(data), numpy.mean(data), numpy.amax(data))
offset_data = offset_mean(data, 0)
print('min, mean, and max of offset data are:', 
      numpy.amin(offset_data),
      numpy.mean(offset_data),
      numpy.amax(offset_data))
```

    original min, mean and max are: 0.0 6.14875 20.0
    min, mean, and max of offset data are: -6.14875 2.842170943040401e-16 13.85125


print('std dev before and after:', numpy.std(data), numpy.std(offset_data))
```

    std dev before and after: 4.613833197118566 4.613833197118566


print('differnece in standard deviation before and after:',
     numpy.std(data) - numpy.std(offset_data))
```

    differnece in standard deviation before and after: 0.0


# offset_mean(data, target_mean_value):
# return a new array containing the original data with its mean offset to match desired value.
# this data should be imported as a measurement in columns in rows
def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```

def offset_mean(data, target_mean_value):
    """Return a new array containing the original data with its mean offset to match the desired value"""
    return(data - numpy.mean(data)) + target_mean_value
```

help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data with its mean offset to match the desired value
    

def offset_mean(data, target_mean_value):
    """Return a new array containing the original data with its mean offset to match the desired value.
    
    Examples
    ----------
    
    >>> Offset_mean([1,2,3], 0)
    array([-1., 0., 1.,])
    """
    return (data - numpy.mean(data)) + target_mean_value


help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data with its mean offset to match the desired value.
        
        Examples
        ----------
        
        >>> Offset_mean([1,2,3], 0)
        array([-1., 0., 1.,])

### LOOPS


odds = [1,3,5,7]
```

print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5
    7

odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5


odds = [1,3,5,7,9,11,13,15,17,19]

for num in odds:
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19


length = 0
names = ['Curie', 'Darwin', 'Turing']
for value in names:
    length = length + 1
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.

name = "Rosalind"
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing


print(len([0,1,2,3]))
```

    4


name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3

### DEFENSIVE PROGRAMMING 

numbers = [1.5, 2.3, 0.7, 0.001, 4.4]
total = 0.0
for num in numbers:
    assert num > 0.0, 'Data should only contain positive values'
    total += num
print('total is:', total)
```

    total is: 8.901


def normalize_rectangle(rect):
    """Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    input should be of the format (x0, y0, x1, x2).
    (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X coordinates'
    assert y0 < y1, 'Invalid Y coordinates'
    
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dx / dy
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx / dy
        upper_x, upper_y = scaled, 1.0
        
    assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
    
    return (0, 0, upper_x, upper_y)
```

print(normalize_rectangle( (0.0, 1.0, 2.0) ))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-3-f9d109085db1> in <module>
    ----> 1 print(normalize_rectangle( (0.0, 1.0, 2.0) ))
    

    <ipython-input-2-06b5b71c9874> in normalize_rectangle(rect)
          3     input should be of the format (x0, y0, x1, x2).
          4     (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    ----> 5     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
          7     assert x0 < x1, 'Invalid X coordinates'


    AssertionError: Rectangles must contain 4 coordinates


print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-4-35750ccfc663> in <module>
    ----> 1 print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0)))
    

    <ipython-input-2-06b5b71c9874> in normalize_rectangle(rect)
          5     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid X coordinates'
          8     assert y0 < y1, 'Invalid Y coordinates'
          9 


    AssertionError: Invalid X coordinates


print(normalize_rectangle( (0.0, 0.0, 1.0, 5.0)))
```

    (0, 0, 0.2, 1.0)


print(normalize_rectangle( (0.0, 0.0, 5.0, 1.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-6-83721d2716e7> in <module>
    ----> 1 print(normalize_rectangle( (0.0, 0.0, 5.0, 1.0)))
    

    <ipython-input-2-06b5b71c9874> in normalize_rectangle(rect)
         18 
         19     assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    ---> 20     assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
         21 
         22     return (0, 0, upper_x, upper_y)


    AssertionError: Calculated upper y coordinate invalid


### TRANSCRITPION


# Prompt user to enter the input fasta file name

input_file_name = input("Enter the name of the input fasta file")
```

# Open the input fasta file and read the DNA sequence.

with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in unput_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip() 

# Transcribe the DNA to RNA
rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide

# Prompt use to enter the output file name

out_file_name = input("Enter the name of the output: ")
```


print(rna_sequence)


### TRANSLATION

# prompt the user to enter input RNA file name

input_file_name - input("Enter the name of the input RNA file:")
```

# Open the input RNA file and read the DNA sequence

with open(input_file_name, "r") as input_file:
    rna_sequence - input_file.read().strip()
```

# Define the codon table

codon_table - {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S", 
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "T",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "A",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", 
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E", 
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W", 
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R", 
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GCU": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}
```


# Translate RNA to protein

protein_sequence = " "
for i in range(0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


# Prompt the user to enter the output file name

output_file_name = input("Enter the name of the output file: ")
```


# Save the protein sequence to a text file

with open(output_file_name, "W") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_file_name}")
    

protein(protein_sequence)
```

### ERRORS

# This code has an intentional error. You can type it directly or use it for reference to understand the error 
# message below

def favorite_ice_cream():
    ice_creams = [
        'chocolate',
        'vanilla',
        'strawberry'
    ]
    print(ice_creams[3])
    
favorite_ice_cream()
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-2-062f04a982e6> in <module>
         10     print(ice_creams[3])
         11 
    ---> 12 favorite_ice_cream()
    

    <ipython-input-2-062f04a982e6> in favorite_ice_cream()
          8         'strawberry'
          9     ]
    ---> 10     print(ice_creams[3])
         11 
         12 favorite_ice_cream()


    IndexError: list index out of range



```python
def some_function():
    msg = 'hello world'
    print(msg)
    return msg

print(a)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-bca0e2660b9f> in <module>
    ----> 1 print(a)
    

    NameError: name 'a' is not defined

print('hello')
```

    hello

count = 0
for number in range(10):
    count = count + number
print('The count is:', count)
```

    The count is: 45

letters = ['a', 'b', 'c']

print('letter #1 is', letters[0])
print('letter #2 is', letters[1])
print('letter #3 is', letters[2])
#print('letter #4 is', letters[3])
```

    letter #1 is a
    letter #2 is b
    letter #3 is c

file_handle = open('myfile.txt', 'w')

