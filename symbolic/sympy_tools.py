import os
import types
import dill
from StringIO import StringIO
from collections import OrderedDict
from sympy import *


def generate_sensitivity_equations(equations, params):
    sens_eqns = OrderedDict()
    for var_i, f_i in equations.items():
        for par_j in params.keys():
            if params[par_j] == 'fixed':
                # We don't calculate sensitivity wrt fixed parameters
                continue
            dsens = diff(f_i, par_j)
            for var_k in equations.keys():
                sens_kj = Symbol('sens%s_%s' % (var_k, par_j))
                dsens += diff(f_i, var_k) * sens_kj

            sens_eqns['sens%s_%s' % (var_i, par_j)] = simplify(dsens)

    return sens_eqns


def generate_model_jacobian(equations):
    jacobian_equations = OrderedDict()
    for var_i, f_i in equations.items():
        for var_j in equations.keys():
            dfi_dj = Symbol('d%s_d%s' % (var_i, var_j))
            jacobian_equations[dfi_dj] = diff(f_i, var_j)

    return jacobian_equations


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


def _write_jit_model(variables, diff_eqns, params, output_fh, sens_eqns=None, jacobian_eqns=None,
                     subexpressions=None, imports=None):
    """ Write out a model to a file-like object"""

    lines = []
    pad = ' ' * 4  # 4 spaces as padding.

    # Import numpy and numba by default.
    default_imports = ["import numpy as np",
                       "from numba import njit"]

    for import_string in default_imports:
        if import_string not in imports:
            imports.append(import_string)

    for import_string in imports:
        lines.append(import_string)
    lines.append("\n")

    # Write out the parameters in the order they appear in the model
    ordered_param_str = ", ".join(["'%s'" % par for par in params.keys()])
    lines.append("ordered_params = [ %s ]" % ordered_param_str)
    lines.append("n_vars = %d" % len(variables))
    lines.append("\n")

    if sens_eqns is None:
        lines.append("@njit")
        lines.append("def model(y, t, yout, p):")
    else:
        lines.append("def sens_model(y, t, yout, p):")

    # ----------------------------------------------------------------------
    # Write out the parameter block.
    # Parameters are contained in p, a 1d numpy array.
    # ----------------------------------------------------------------------
    lines.append("")
    lines.append("#*! Parameters Start")
    lines.append(pad + "#---------------------------------------------------------#")
    lines.append(pad + "#Parameters#")
    lines.append(pad + "#---------------------------------------------------------#\n")
    for idx, par in enumerate(params):
        lines.append(pad + "%s = p[%d]" % (par, idx))
    lines.append("#*! Parameters End")

    # ----------------------------------------------------------------------
    # Write out the state variables block.
    # Variables are contained in y, a 1d numpy array.
    # ----------------------------------------------------------------------
    lines.append("\n")
    lines.append("#*! Variables Start")
    lines.append(pad + "#---------------------------------------------------------#")
    lines.append(pad + "#Variables#")
    lines.append(pad + "#---------------------------------------------------------#\n")
    for idx, var in enumerate(variables):
        lines.append(pad + "%s = y[%d]" % (var, idx))

    # if the model contains sensitivity equations, we have an extra pxn state variables
    if sens_eqns is not None:
        lines.append("\n")
        lines.append(pad + "#---------------------------------------------------------#")
        lines.append(pad + "#Sensitivity Variables#")
        lines.append(pad + "#---------------------------------------------------------#\n")
        total_vars = len(variables)
        for idx, var in enumerate(sens_eqns.keys()):
            lines.append(pad + "%s = y[%d]" % (var, idx + total_vars))
    lines.append("#*! Variables End")

    # ----------------------------------------------------------------------
    # Write out the subexpressions block.
    # ----------------------------------------------------------------------
    if subexpressions is not None:
        lines.append("\n")
        lines.append("#*! Subexpressions Start")
        lines.append(pad + "#---------------------------------------------------------#")
        lines.append(pad + "#Subexpressions#")
        lines.append(pad + "#---------------------------------------------------------#\n")
        for expr_var, expr in enumerate(subexpressions.items()):
            lines.append(pad + "%s = %s" % (expr[0], expr[1]))
    lines.append("#*! Subexpressions End")

    # ----------------------------------------------------------------------
    # Write out the actual differential equations.
    # We store their values in yout, a 1d numpy array
    # ----------------------------------------------------------------------
    lines.append("\n")
    lines.append("#*! Differential Equations Start")
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
    lines.append("#*! Differential Equations End")

    if jacobian_eqns is not None:
        all_eqns = diff_eqns.values()
        if sens_eqns is not None:
            all_eqns += sens_eqns.values()

        lines.append('\n\n')
        lines.append("@njit")
        lines.append("def model_jacobian(y, t, yout, p):")
        lines.append("")

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

        lines.append("\n")
        if sens_eqns is not None:
            lines.append("\n")
            lines.append(pad + "#---------------------------------------------------------#")
            lines.append(pad + "#sensitivity Variables#")
            lines.append(pad + "#---------------------------------------------------------#\n")
            total_vars = len(variables)
            for idx, var in enumerate(sens_eqns.keys()):
                lines.append(pad + "%s = y[%d]" % (var, idx + total_vars))

        lines.append("\n")
        i = j = 0
        for jac_var, eqn in jacobian_eqns.items():
            eqn_str = "%s" % eqn
            if eqn_str != "0":
                lines.append(pad + "yout[%d,%d] = (%s) #  %s" % (i, j, eqn, jac_var))
            else:
                lines.append(pad + "# yout[%d,%d] = (%s) #  %s" % (i, j, eqn, jac_var))
            i += 1
            if i >= len(all_eqns):
                i = 0
                j += 1
                lines.append("")

    for line in lines:
        output_fh.write(line + "\n")
        # Actually write out the model


def parse_model_file(model):
    """

    :param model: StringIO | str | types.FunctionType
    :return:
    :rtype: dict
    """

    if types.FunctionType == type(model):
        model_text = dill.source.getsource(model)
    else:
        if type(model) == str:
            model = open(model, 'r').read()
        model_text = model.read()
        model.close()

    categories = {'Parameters': False, 'Variables': False,
                  'Conservation Laws': True, 'Rate Laws': True,
                  'Differential Equations': True, 'Imports': False}

    parsed_model = {}

    for category, sympify_rhs in categories.items():

        # Each chunk starts with those markers.
        start_idx = model_text.find("#*! %s Start" % category)
        end_idx = model_text.find("#*! %s End" % category)
        chunk = model_text[start_idx:end_idx].split('\n')

        # Imports are just a raw string we copy and paste at the start of the file.
        if category == 'Imports':
            symbols_dict = chunk
        else:
            symbols_dict = _process_chunk(chunk, sympify_rhs)

        parsed_model[category] = symbols_dict

    return parsed_model


def make_jacobian(model, output_file=None, simplify_subexpressions=False):
    model_dict = parse_model_file(model)

    eqns = model_dict['Differential Equations']
    rate_laws = model_dict['Rate Laws']
    cons_laws = model_dict['Conservation Laws']
    variables = model_dict['Variables']
    params = model_dict['Parameters']
    imports = model_dict['Imports']

    model_jac_eqns = generate_model_jacobian(all_eqns)
    model_dict['Model Jacobian Equations'] = model_jac_eqns


def make_jit_model(model, output_file=None, fixed_params=None, make_model_sensitivities=True,
                   simplify_subexpressions=False, make_model_jacobian=False):
    """

    :param model: str | StringIO | func
    :param output_file:
    :param fixed_params:
    :param make_model_sensitivities:
    :param simplify_subexpressions:
    :param make_model_jacobian:
    :return:
    """

    model_dict = parse_model_file(model)

    eqns = model_dict['Differential Equations']
    rate_laws = model_dict['Rate Laws']
    cons_laws = model_dict['Conservation Laws']
    variables = model_dict['Variables']
    params = model_dict['Parameters']
    imports = model_dict['Imports']

    if fixed_params is not None:
        for f_p in fixed_params:
            try:
                params[f_p] = 'fixed'
            except KeyError:
                raise KeyError('%s not in model parameters' % f_p)

    expanded_eqns = OrderedDict()
    for d_var, eqn in eqns.items():
        expanded_eqns[d_var[1:]] = eqn.subs(rate_laws).subs(cons_laws).simplify()

    # Sensitivity Equations
    sens_eqns = None
    if make_model_sensitivities:
        sens_eqns = generate_sensitivity_equations(expanded_eqns, params)
    model_dict['Sensitivity Equations'] = sens_eqns

    all_eqns = OrderedDict()
    for d_var, eqn in expanded_eqns.items():
        all_eqns[d_var] = eqn

    if sens_eqns is not None:
        for d_var, eqn in sens_eqns.items():
            all_eqns[d_var] = eqn

    # Model Jacobian
    model_jac_eqns = None
    if make_model_jacobian:
        model_jac_eqns = generate_model_jacobian(all_eqns)
    model_dict['Model Jacobian Equations'] = model_jac_eqns

    # Subexpressions
    subexpressions = None
    if simplify_subexpressions:

        repeated_exps, simplified_eqns = cse(all_eqns.values(), optimizations='basic')
        subexpressions = OrderedDict(repeated_exps)

        for d_var, simp_eqn in zip(expanded_eqns, simplified_eqns[:len(eqns)]):
            expanded_eqns[d_var] = simp_eqn

        if sens_eqns is not None:
            for d_var, simp_eqn in zip(sens_eqns, simplified_eqns[len(eqns):]):
                sens_eqns[d_var] = simp_eqn

        model_dict['Differential Equations'] = simplified_eqns[:len(eqns)]
        model_dict['Sensitivity Equations'] = simplified_eqns[len(eqns):]

    model_dict['Subexpressions'] = subexpressions

    _write_jit_model(variables, expanded_eqns, params, output_file, sens_eqns, model_jac_eqns,
                     subexpressions, imports)

    return model_dict
