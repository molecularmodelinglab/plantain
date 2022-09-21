
def safe_index(list, item, debug=False):
    """ Taken from the equibind code -- basically, just make sure we don't throw
    an error if e.g. some weird hybridization occurs on our mol """
    if item in list:
        return list.index(item)
    else:
        if debug:
            raise AssertionError()
        return len(list) - 1

def dict_to_id_str(d):
    """ returns a compressed string representation of a dict.
    Note that it is not meant to be bidirectional; you can't
    read back the dict from the string. But it's useful for a
    "readable dict hash" for caching files for a particular 
    config """
    arr = []
    for key in sorted(d.keys()):
        assert isinstance(key, str)
        val = d[key]
        if isinstance(val, dict):
            val_str = "(" + dict_to_id_str(val) + ")"
        else:
            val_str = str(val).replace(' ', '').replace('[', '(').replace(']', ')')
        arr.append(f"{key}={val_str}")
    ret = ",".join(arr)
    for c in "{}[] \t":
        assert c not in ret
    return ret