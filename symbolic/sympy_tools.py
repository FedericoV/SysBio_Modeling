import os
import cPickle as pickle
from collections import OrderedDict
import imp

from sympy import *


def _process_chunk(chunk, sympify_rhs=False):
    symbols_dict = OrderedDict()

    for line in chunk:
        line = line.replace(" ", "")  # Spaces
        line = line.replace("\t", "")  # Tabs

        if line.startswith("#") or line == "":
            continue

        lhs = line[:line.find("=")]
        rhs = line[line.find("=") + 1:]
        if not sympify_rhs:
            symbols_dict[lhs] = Symbol(lhs)
        else:
            symbols_dict[lhs] = sympify(rhs)

    return symbols_dict


def parse_model_file(model_fh):
    model_text = model_fh.read()

    categories = {'Parameters': False, 'Variables': False,
                  'Conservation Laws': True, 'Rate Laws': True,
                  'Differential Equations': True}

    parsed_model_dict = {}

    for category, sympify_rhs in categories.items():
        start_idx = model_text.find("#*! %s Start" % category)
        end_idx = model_text.find("#*! %s End" % category)

        chunk = model_text[start_idx:end_idx].split('\n')
        symbols_dict = _process_chunk(chunk, sympify_rhs)

        parsed_model_dict[category] = symbols_dict

    bound_start_idx = model_text.find("#*! Bound Arguments Start")
    bound_end_idx = model_text.find("#*! Bound Arguments End")
    bound_args = model_text[bound_start_idx:bound_end_idx].split("\n")
    parsed_model_dict["Bound Arguments"] = bound_args
    return parsed_model_dict


def write_model(variables, diff_eqns, bound_args, output_fh, sens_eqns=None,
                model_name="sensitivity_model"):
    lines = []
    pad = "    "
    dpad = pad + pad

    lines.append("def make_bound_model(*args):")

    for line in bound_args:
        lines.append(line)
    lines.append("\n")

    lines.append(pad + "def %s(y, t):" % model_name)

    lines.append("\n")
    lines.append(dpad + "#---------------------------------------------------------#")
    lines.append(dpad + "#Variables#")
    lines.append(dpad + "#---------------------------------------------------------#\n")
    for idx, var in enumerate(variables):
        lines.append(dpad + "%s = y[%d]" % (var, idx))

    if sens_eqns is not None:
        lines.append("\n")
        lines.append(dpad + "#---------------------------------------------------------#")
        lines.append(dpad + "#sensitivity Variables#")
        lines.append(dpad + "#---------------------------------------------------------#\n")
        total_vars = len(variables)
        for idx, var in enumerate(sens_eqns.keys()):
            lines.append(dpad + "%s = y[%d]" % (var, idx + total_vars))

    lines.append("\n")
    lines.append(dpad + "#---------------------------------------------------------#")
    lines.append(dpad + "#Differential Equations#")
    lines.append(dpad + "#---------------------------------------------------------#\n")

    return_vars = []
    for var, eqn in diff_eqns.items():
        lines.append(dpad + "d%s = (%s)" % (var, eqn))
        return_vars.append("d%s" % var)

    if sens_eqns is not None:
        lines.append("\n")
        lines.append(dpad + "#---------------------------------------------------------#")
        lines.append(dpad + "#sensitivity Equations#")
        lines.append(dpad + "#---------------------------------------------------------#\n")

        for var, eqn in sens_eqns.items():
            lines.append(dpad + "d_%s = (%s)" % (var, eqn))
            return_vars.append("d_%s" % var)

    lines.append("\n")
    lines.append(dpad + "return (%s)" % (", ".join(return_vars)))

    lines.append("\n")
    lines.append(pad + "return %s" % model_name)

    for line in lines:
        output_fh.write(line + "\n")


def write_latex_file(model_fh, output_fh, extended=True):
    '''Currently incomplete'''
    model_dict = parse_model_file(model_fh)
    model_fh.close()

    eqns = model_dict['Differential Equations']
    rate_laws = model_dict['Rate Laws']
    cons_laws = model_dict['Conservation Laws']
    variables = model_dict['Variables']
    bound_args = model_dict['Bound Arguments']
    parameters = model_dict['Parameters']

    new_param_names = {}
    for i, old_name in enumerate(parameters.keys()):
        new_param_names[old_name] = 'k%d' % i

    new_var_names = {}
    for old_name in variables:
        new_var_names[old_name] = old_name[1:]

    expanded_eqns = OrderedDict()
    if extended:
        for d_var, eqn in eqns.items():
            var = Symbol(d_var[2:])
            tmp_eqn = eqn.subs(rate_laws).subs(cons_laws).subs(new_var_names)
            tmp_eqn = tmp_eqn.subs(new_param_names).simplify()
            expanded_eqns[var] = tmp_eqn
    else:
        expanded_eqns = eqns

    output_fh.write('\\documentclass{article}\n')
    output_fh.write('\\usepackage{amsmath}\n')
    output_fh.write('\\allowdisplaybreaks[1]\n')
    output_fh.write('\\begin{document}\n')
    output_fh.write('\\begin{align}\n')
    dt = Symbol('dt')
    for var, eqn in expanded_eqns.items():
        rhs = latex(eqn, mode='plain')
        lhs = latex((var / dt), mode='plain', mul_symbol=None)
        output_fh.write('%s &= %s \\\\ \n\n' % (lhs, rhs))

    output_fh.write('\\end{align}\n')
    output_fh.write('\\end{document}\n')
    output_fh.close()


def make_sensitivity_model(model_fh, sens_model_fh=None, ordered_params=None,
                           calculate_sensitivities=True):
    if 'module' in str(type(model_fh)):
        import inspect

        model_fh = open(inspect.getsourcefile(model_fh))

    if sens_model_fh is None:
        model_fn = os.path.realpath(model_fh.name)
        sens_path = os.path.join(os.path.dirname(model_fn), 'sensitivity')
        if os.path.exists(sens_path):
            sens_model_fh = open(os.path.join(sens_path, 'sensitivity_%s' % os.path.basename(model_fn)), 'w')
        else:
            raise ValueError('No valid output path or sensitivity subdirectory')

    model_dict = parse_model_file(model_fh)
    model_fh.close()

    eqns = model_dict['Differential Equations']
    rate_laws = model_dict['Rate Laws']
    cons_laws = model_dict['Conservation Laws']
    variables = model_dict['Variables']
    bound_args = model_dict['Bound Arguments']

    if ordered_params is None:
        parameters = model_dict['Parameters']
    else:
        parameters = ordered_params
        model_dict['Parameters'] = ordered_params

    expanded_eqns = OrderedDict()
    for d_var, eqn in eqns.items():
        expanded_eqns[d_var[1:]] = eqn.subs(rate_laws).subs(cons_laws).simplify()

    if calculate_sensitivities:
        model_dict['Sensitivity Equations'] = {}

        sens_eqns = OrderedDict()
        for var_i, f_i in expanded_eqns.items():
            for par_j in parameters.keys():
                dsens = diff(f_i, par_j)
                for var_k in expanded_eqns.keys():
                    sens_kj = Symbol('sens%s_%s' % (var_k, par_j))
                    dsens += diff(f_i, var_k) * sens_kj

                sens_eqns['sens%s_%s' % (var_i, par_j)] = simplify(dsens)
        model_dict['Sensitivity Equations'] = sens_eqns
    else:
        sens_eqns = None

    write_model(variables, expanded_eqns, bound_args, sens_model_fh,
                sens_eqns)

    return model_dict