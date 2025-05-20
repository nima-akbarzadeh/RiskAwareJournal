import numpy
from whittle import *
from Markov import *
import warnings
warnings.filterwarnings("ignore")

# # ----------- Finite

# nt, ns, nc, ut, th, fr = 5, 5, 5, (1, 0), 0.9, 0.3
# na = nc * ns

# rew_vals = rewards(nt, na, ns)
# prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
# markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

# RiskAware_Whittle = RiskAwareWhittle(ns, na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
# RiskAware_Whittle.get_indices(nt, nt*ns*na)

# print(np.max(RiskAware_Whittle.whittle_indices))

# # ----------- Nonstationry

for nt in [10]:
    for df in [0.9]:
        for ns in [3]:
            for ng in [50]:
                for nc in [3]:
                    for ut in [(1, 0), (2, 4), (3, 16)]:
                        for th in [0.5, 0.9]:
                            for fr in [0.1]:

                                key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nc{nc}_ut{ut}_th{th}_fr{fr}'
                                na = nc * ns

                                rew_vals = rewards_ns(df, nt, na, ns)
                                prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
                                markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

                                RiskAware_Whittle = RiskAwareWhittleNS([ns, ng], na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
                                RiskAware_Whittle.get_indices(ng, ng*ns*na)
                                print(f"{key_value}: {[print(np.max(RiskAware_Whittle.whittle_indices[a])) for a in range(na)]}")
                                # RiskAware_Whittle.backward_discreteliftedstate(0, 0)


# # ----------- Infinite

# df, nt, ns, ng, nc, ut, th, fr = 0.9, 10, 2, 10, 2, (2, 8), 0.5, 0.3
# na = nc * ns

# rew_vals = rewards_inf(na, ns)
# prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
# markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

# RiskAware_Whittle = RiskAwareWhittleInf([ns, ng, ng], na, rew_vals, markov_matrix, df, nt, ut[0], ut[1], th)
# RiskAware_Whittle.get_indices(2*ng, ng*ns*na)