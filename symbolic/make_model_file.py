import os
import sys
import inspect
from SysBio_Modeling.symbolic import make_jit_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sens", help="writes sensitivity equations",
                    action="store_true")
parser.add_argument("-j", "--jac", help="writes the jacobian of the state variables",
                    action="store_true")
parser.add_argument("-i", "--input", help="input model file",
                    required=True, type=str)
parser.add_argument("-o", "--output", help="output file",
                    required=False, type=str)
parser.add_argument("-fp", "--fixed_params", help="fixed parameters in the model",
                    required=False, type=str)

args = parser.parse_args()

if not os.path.exists(args.input):
    raise ValueError('Input file %s does not exist.' % model_file)
model_fh = open(args.input, 'r')

if args.output is not None:
    jit_model_fh = open(args.output, 'w')
else:
    jit_model_fh = None

output = make_jit_model(model_fh, jit_model_fh, make_model_sensitivities=False, make_model_jacobian=True,
                        simplify_subexpressions=False)
jit_model_fh.close()

jit_sens_model_name = "sensitivity_" + model_name + "_simplified_jit.py"
jit_sens_model_name_fh = open(os.path.join(model_dir, 'sensitivity', jit_sens_model_name), 'w')
make_jit_model(base_model, jit_sens_model_name_fh, fixed_params=['gal', 're_total', 'n'],
               make_model_sensitivities=True, simplify_subexpressions=False, make_model_jacobian=True)
jit_sens_model_name_fh.close()
