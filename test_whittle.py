import numpy
from whittle import *
from Markov import *
import warnings
warnings.filterwarnings("ignore")

df, nt, ns, ng, nc, ut, th, fr = 0.9, 10, 2, 10, 2, (2, 8), 0.5, 0.3
na = nc * ns

rew_vals = rewards_inf(na, ns)
prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

RiskAware_Whittle = RiskAwareWhittleInf([ns, ng, ng], na, rew_vals, markov_matrix, df, nt, ut[0], ut[1], th)
RiskAware_Whittle.get_indices(2*ng, ng*ns*na)