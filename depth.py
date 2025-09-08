from core.interface import DepthInterface

from circuits.circuit import *

interface = DepthInterface()

n = 32

#print(setup_full_adder(interface, bit_len=n))
#print(setup_ripple_carry_adder(interface, bit_len=n))
#print(setup_carry_look_ahead_adder(interface, bit_len=n))
#print(setup_wallace_tree_multiplier(interface, bit_len=n))
#print(setup_theorem_4_2(interface, bit_len=n))
#print(setup_theorem_5_2(interface, bit_len=n))
#print(setup_theorem_5_2_step_3(interface, bit_len=n)) # - 16
print(setup_theorem_5_2_step_4(interface, bit_len=n))