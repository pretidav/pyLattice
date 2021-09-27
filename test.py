from linalg.tensors import *
    

a  = Real(10)
b  = Real(-3)
c  = 10 
ac = Complex(3+5j)
bc = Complex(2-8j)

#Real tests 
def test_sum_real_real():
    assert( (a+b).value  ==  7)
def test_diff_real_real():
    assert( (a-b).value  ==  13)
def test_mul_real_real():
    assert( (a*b).value  ==  -30)
def test_mul_real_float():
    assert( (a*c).value  ==  100)
def test_pow_real_float():
    assert( (a**c).value ==  10000000000)
def test_pow_real_real():
    assert( (a**b).value ==  0.001)
def test_mul_real_complex():
    assert((a*bc).value  == 20-80j)

#Complex tests
def test_sum_complex_complex():
    assert((ac+bc).value == 5-3j)
def test_diff_complex_complex():
    assert((ac-bc).value == 1+13j)
def test_mul_complex_complex():
    assert((ac*bc).value == 46-14j)
def test_mul_complex_real():
    assert((ac*a).value  == 30+50j)
