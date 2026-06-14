from Function.Symbols import *


def get_combined_subs_2D(noise):
    if not noise:
        subs = {
            D21 : 0,
            D31 : - D11,
            D12 : D22 - D32,
            D13 : D23 - D33,
            D22: D23,
            D23: 1 / (E11 + E21 + E31),
            E12 : -(E22 + E32),
            E13: -E23 - E33,
        }
    else:
        subs = {
            D11: 0, D13: 0,
            D31: 0,
            D33: 0,
            D21: 0,
            D23: 0,

            E11: -(E21 + E31),
            E12: -(E22 + E32),
            E13: -(E23 + E33),

            D12: -(D22 + D32),
          #  D22: - D23/2,
          #  D21: D23
        }
     #      D11: D31,
     #      D12: D22 - D32,
     #      D13: D33,
     #      D22: D23,
     #      E11: E31,
     #      E12: E32,
     #      E13: E33,
     #
     #  }
    #    subs = {
     #        D12: D22 - D32,
        #     D21 : 2 / (E12 + 2*E13 + E22 + 2*E23 + E32 + 2*E33) ,
      #      2 / (E13 + E21 + E23 + E31 + E32) : D23,
      #      1 / (E13 + E21 + 2*E22 + E23 + E33 + 2*E32) : D11,
      #      1 / (E21 + E31 + E33) : D22
     #   }
    return subs


# subs = {
#     D12: D22 - D32,
#     D13: D33,
#     D31: (D12 * D13) / D23
# }

def get_combined_subs_1D(choose):
    subs = {}
    if choose == 0:
        subs = {
            E1 : 1 / (D2 * (sigma2 + 1)),
            E2 : 0,
            E3: 0,
            D1: 0,
            D3: D2
        }
    elif choose == 1:
        subs = {
                E1: 0,
                E2: 1 / (D2 * (sigma2 + 1)),
                E3: 0,
                D1: D2,
                D3: 0
            }
    return subs


def get_1D_sum_subs(w0):
    if w0 % 2 == 0:
        return {
            E2: 0,
            E1: 0,
            D2: 0,
            E3: 0,
            D3: 0,
            D1: 0
        }
    else:
        return {
            E1: 1 / (D2 * (sigma2 + 1)),
            E2: 2 / (D2 * (sigma2 + 2)),
            E3: -sigma2 / (D2 * (sigma2 + 1) * (sigma2 + 2)),
            D1: 0,
            D3: 0
        }

def get_2D_sum_subs(w0):
    if w0 % 2 == 0:
        return {
            D11: 0, D13: 0, D21: 0, D23: 0, D31: 0, D33: 0,
            D22: -(D21 + D23),
            E11: -(E31 + E21),
            E32: -(E12 + E22 + E13 + E23 + E33)
        }
    else:
        return {
            D11: 0, D13: 0, D31: 0, D33: 0,
            D32: -D12,
            E11: -(E31 + E21),
            E32: -(E12 + E22 + E13 + E23 + E33)
        }

def get_generic_subs(use_1D, tpe, func_instance):
    # For cases where we simply zero out all entries.
    if use_1D:
        if tpe == "L":
            # Assuming func_instance.E and func_instance.D are iterables of symbols
            subs_dict = {e: 0 for e in func_instance.E}
            subs_dict.update({d: 0 for d in func_instance.D})
            return subs_dict
        else:
            subs_dict = {e: 0 for e in func_instance.E}
            subs_dict.update({d: 0 for d in func_instance.D})
            return subs_dict
    else:
        if tpe == "L":
            subs_dict = {e: 0 for e in func_instance.E}
            subs_dict.update({d: 0 for d in func_instance.D})
            return subs_dict
        else:
            subs_dict = {e: 0 for e in func_instance.E}
            subs_dict.update({d: 0 for d in func_instance.D})
            return subs_dict

