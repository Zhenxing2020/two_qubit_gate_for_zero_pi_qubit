import sys
sys.path.append('../')
import utils_2Q_gate_zp as ut




ut.compare_two_lists(ut.truc_model['cnot_82'], ut.truc_model['cnot_short_500'][:200])
ut.compare_two_lists(ut.truc_model['cnot_82'], ut.truc_model['cnot_short_1000'][:200])

