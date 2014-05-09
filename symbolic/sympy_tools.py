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

    return parsed_model_dict


def write_jit_model(variables, diff_eqns, params, output_fh,
                    sens_eqns=None, write_ordered_params=False):
    lines = []
    pad = "    "

    if write_ordered_params:
        ordered_param_str = ", ".join(["'%s'" %par for par in params.keys()])
        lines.append("ordered_params = [ %s ]" %ordered_param_str)
        lines.append("n_vars = %d" % len(variables))
        lines.append("\n")

    if sens_eqns is None:
        lines.append("def model(y, t, yout, p):")
    else:
        lines.append("def sens_model(y, t, yout, p):")

    lines.append("\n")
    lines.append(pad + "#---------------------------------------------------------#")
    lines.append(pad + "#Parameters#")
    lines.append(pad + "#---------------------------------------------------------#\n")
    for idx, var in enumerate(params):
        lines.append(pad + "%s = p[%d]" % (var, idx))

    lines.append("\n")
    lines.append(pad + "#---------------------------------------------------------#")
    lines.append(pad + "#Variables#")
    lines.append(pad + "#---------------------------------------------------------#\n")
    for idx, var in enumerate(variables):
        lines.append(pad + "%s = y[%d]" % (var, idx))

    if sens_eqns is not None:
        lines.append("\n")
        lines.append(pad + "#---------------------------------------------------------#")
        lines.append(pad + "#sensitivity Variables#")
        lines.append(pad + "#---------------------------------------------------------#\n")
        total_vars = len(variables)
        for idx, var in enumerate(sens_eqns.keys()):
            lines.append(pad + "%s = y[%d]" % (var, idx + total_vars))

    lines.append("\n")
    lines.append(pad + "#---------------------------------------------------------#")
    lines.append(pad + "#Differential Equations#")
    lines.append(pad + "#---------------------------------------------------------#\n")

    for idx, eqn in enumerate(diff_eqns.values()):
        lines.append(pad + "yout[%d] = (%s)" % (idx, eqn))

    if sens_eqns is not None:
        lines.append("\n")
        lines.append(pad + "#---------------------------------------------------------#")
        lines.append(pad + "#sensitivity Equations#")
        lines.append(pad + "#---------------------------------------------------------#\n")

        for idx, eqn in enumerate(sens_eqns.values()):
            lines.append(pad + "yout[%d] = (%s)" % (idx + total_vars, eqn))

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


def make_sensitivity_model(model_fh, sens_model_fh=None, fixed_params=None,
                           calculate_sensitivities=True, write_ordered_params=None):
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
    params = model_dict['Parameters']

    if fixed_params is not None:
        for f_p in fixed_params:
            if f_p in params:
                params[f_p] = 'fixed'
            else:
                print '%s not in model parameters' % f_p

    expanded_eqns = OrderedDict()
    for d_var, eqn in eqns.items():
        expanded_eqns[d_var[1:]] = eqn.subs(rate_laws).subs(cons_laws).simplify()

    if calculate_sensitivities:
        model_dict['Sensitivity Equations'] = {}

        sens_eqns = OrderedDict()
        for var_i, f_i in expanded_eqns.items():
            for par_j in params.keys():
                if params[par_j] == 'fixed':
                    # We don't calculate sensitivity wrt fixed parameters
                    continue
                dsens = diff(f_i, par_j)
                for var_k in expanded_eqns.keys():
                    sens_kj = Symbol('sens%s_%s' % (var_k, par_j))
                    dsens += diff(f_i, var_k) * sens_kj

                sens_eqns['sens%s_%s' % (var_i, par_j)] = simplify(dsens)
        model_dict['Sensitivity Equations'] = sens_eqns
    else:
        sens_eqns = None

    if write_ordered_params is None:
        if calculate_sensitivities is True:
            write_ordered_params = False
        else:
            write_ordered_params = True

    write_jit_model(variables, expanded_eqns, params, sens_model_fh,
                    sens_eqns, write_ordered_params)

    return model_dict