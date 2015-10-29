import types
import dill
from collections import OrderedDict
from sympy import *
import numpy as np


def wrap_model(yout_size, model):
    yout = np.zeros(yout_size)

    def wrapped_model(y, t, p):
        model(y, t, yout, p)
        return yout

    return wrapped_model


def _sympify_chunk(chunk, sympify_rhs=False):
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


def _write_header(model_dict, fcn_name):
    lines = []
    pad = ' ' * 4  # 4 spaces as padding.

    subexpressions = model_dict['Subexpressions']
    params = model_dict['Parameters']
    imports = model_dict['Imports']
    sens_eqns = model_dict['Sensitivity Equations']
    variables = model_dict['Variables']

    # ----------------------------------------------------------------------
    # Write Imports
    # ----------------------------------------------------------------------
    # Import numpy and numba by default.
    default_imports = ["import numpy as np",
                       "from numba import njit"]

    if imports is None:
        imports = default_imports

    else:
        for import_string in default_imports:
            if import_string not in imports:
                imports.append(import_string)

    for import_string in imports:
        lines.append(import_string)
    lines.append("\n")

    # ----------------------------------------------------------------------
    # Write out the parameters, variable names, and function header
    # ----------------------------------------------------------------------
    ordered_param_str = ", ".join(["'%s'" % par for par in params.keys()])
    lines.append("ordered_params = [ %s ]" % ordered_param_str)
    lines.append("n_vars = %d" % len(variables))
    lines.append("\n")

    lines.append("@njit")
    lines.append("def %s(y, t, yout, p):" % fcn_name)

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
        n_state_vars = len(variables)
        for idx, var in enumerate(sens_eqns.keys()):
            lines.append(pad + "%s = y[%d]" % (var, idx + n_state_vars))
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

    return lines


def _derive_sensitivity_equations(equations, params):
    sens_eqns = OrderedDict()
    for var_i, f_i in equations.items():
        for par_j in params.keys():
            if params[par_j] == 'fixed':
                # We don't calculate sensitivity wrt fixed parameters
                continue
            dsens = diff(f_i, par_j)
            for var_k in equations.keys():
                sens_kj = Symbol('sens_%s_%s' % (var_k, par_j))
                dsens += diff(f_i, var_k) * sens_kj

            sens_eqns['d_sens_%s_%s' % (var_i, par_j)] = simplify(dsens)
    return sens_eqns


def _derive_jacobian_equations(equations):
    jacobian_equations = OrderedDict()
    for var_i, f_i in equations.items():
        for var_j in equations.keys():
            dfi_dj = Symbol('d_%s_d_%s' % (var_i, var_j))
            jacobian_equations[dfi_dj] = diff(f_i, var_j)

    return jacobian_equations


def make_ode_model(model_dict, output_fh=None):
    """ Write out a model to a file-like object"""

    sens_eqns = model_dict['Sensitivity Equations']
    diff_eqns = model_dict['Differential Equations']

    n_total_vars = len(diff_eqns)
    if sens_eqns:
        n_total_vars += len(diff_eqns)

    pad = ' ' * 4  # 4 spaces as padding.
    lines = _write_header(model_dict, fcn_name='model')

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

    if sens_eqns:
        lines.append("\n")
        lines.append(pad + "#---------------------------------------------------------#")
        lines.append(pad + "#sensitivity Equations#")
        lines.append(pad + "#---------------------------------------------------------#\n")

        for idx, eqn in enumerate(sens_eqns.values()):
            lines.append(pad + "yout[%d] = (%s)" % (idx + len(diff_eqns), eqn))
    lines.append("#*! Differential Equations End")

    # Actually write out the model
    if output_fh:
        for line in lines:
            output_fh.write(line + "\n")

    # Return a function-like object
    fcn_str = "\n".join(lines)

    # puts 'model' fcn in scope
    exec fcn_str

    # wrap model:
    # noinspection PyUnresolvedReferences
    wrapped_model = wrap_model((n_total_vars,), model)

    return wrapped_model


def make_ode_model_jacobian(model_dict, output_fh=None):
    model_jac_eqns = model_dict['Model Jacobian Equations']
    sens_eqns = model_dict['Sensitivity Equations']
    diff_eqns = model_dict['Differential Equations']

    n_total_vars = len(diff_eqns)
    if sens_eqns:
        n_total_vars += len(sens_eqns)

    pad = ' ' * 4  # 4 spaces as padding.
    # very important that the string 'model_jacobian' matches the returned
    # variable at the bottom due to dynamic code injection via exec
    lines = _write_header(model_dict, fcn_name='model_jacobian')

    # ----------------------------------------------------------------------
    # Write out the jacobian equations block.
    # Outputs are stored in yout, a 2d numpy array.
    # ----------------------------------------------------------------------
    lines.append("\n")
    i = j = 0
    for jac_var, eqn in model_jac_eqns.items():
        eqn_str = "%s" % eqn
        if eqn_str != "0":
            lines.append(pad + "yout[%d,%d] = (%s) #  %s" % (i, j, eqn, jac_var))
        else:
            lines.append(pad + "# yout[%d,%d] = (%s) #  %s" % (i, j, eqn, jac_var))
        i += 1
        if i >= n_total_vars:
            i = 0
            j += 1
            lines.append("")

    # Actually write out the model
    if output_fh:
        for line in lines:
            output_fh.write(line + "\n")

    # Return a function-like object
    fcn_str = "\n".join(lines)

    # puts 'model_jacobian' fcn in scope
    exec fcn_str

    # wrap model:
    # noinspection PyUnresolvedReferences
    wrapped_model = wrap_model((n_total_vars, n_total_vars), model_jacobian)
    # noinspection PyUnresolvedReferences
    return wrapped_model


def parse_model_file(model):
    """

    :param model: StringIO | str | types.FunctionType
    :return:
    :rtype: dict
    """
    if 'numba.targets.registry.CPUOverloaded' in str(type(model)):
        model = model.py_func

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
            symbols_dict = _sympify_chunk(chunk, sympify_rhs)

        parsed_model[category] = symbols_dict

    return parsed_model


def process_model_dict(model_dict, fixed_params=None, calculate_model_sensitivities=True,
                       simplify_subexpressions=False, calculate_model_jacobian=False):
    """

    :param model: str | StringIO | func
    :param output_file:
    :param fixed_params:
    :param make_model_sensitivities:
    :param simplify_subexpressions:
    :param make_model_jacobian:
    :return:
    """

    eqns = model_dict['Differential Equations']
    rate_laws = model_dict['Rate Laws']
    cons_laws = model_dict['Conservation Laws']
    params = model_dict['Parameters']

    if fixed_params is not None:
        for f_p in fixed_params:
            try:
                params[f_p] = 'fixed'
            except KeyError:
                raise KeyError('%s not in model parameters' % f_p)

    # Expand out the differential equations and substitute in all conservation laws
    # and auxiliary expressions.
    expanded_eqns = OrderedDict()
    for d_var, eqn in eqns.items():
        expanded_eqns[d_var[2:]] = eqn.subs(rate_laws).subs(cons_laws).simplify()

    # Derive the sensitivity equations
    sens_eqns = None
    if calculate_model_sensitivities:
        sens_eqns = _derive_sensitivity_equations(expanded_eqns, params)
    model_dict['Sensitivity Equations'] = sens_eqns

    all_eqns = OrderedDict()
    for d_var, eqn in expanded_eqns.items():
        all_eqns[d_var] = eqn

    if sens_eqns is not None:
        for d_var, eqn in sens_eqns.items():
            all_eqns[d_var] = eqn

    # Model Jacobian
    model_jac_eqns = None
    if calculate_model_jacobian:
        model_jac_eqns = _derive_jacobian_equations(all_eqns)
    model_dict['Model Jacobian Equations'] = model_jac_eqns

    # Simplifications:
    # We use SymPy to find common subexpressions that occur multiple times in all the equations
    # and pre-compute them.  Results in a non-trivial speed up but can make it harder to see
    # original equations.
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
    return model_dict
