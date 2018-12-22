Wrap_N_c = 1


# ============================================================


# auto wrap for long list in graphviz
def wrap_str(name, maxlen=float('inf'), width=float('inf'), trimer=False):
    # if input is str: split into list
    if isinstance(name, str):
        long_list = name.split()
    else:
        long_list = name

    # if all elements are int => use int abbreviation (m-n)
    for n in long_list:
        if not n.isdigit():
            break
    else:
        if len(long_list) > 2:
            long_list = int_wrap(long_list)

    # for LHCII trimer: A1-A5, B3-B5 ....
    if trimer:
        first_alphabet_dict = {}
        # drop A, B, C
        for n in long_list:
            if n[0] in first_alphabet_dict:
                first_alphabet_dict[n[0]].append(n[1:])
            else:
                first_alphabet_dict[n[0]] = [n[1:]]
        # re-add A, B, C
        long_list = []
        for k, v in first_alphabet_dict.items():
            if len(v) > 2:
                int_wrapped = int_wrap(v)
                for j, s in enumerate(int_wrapped):
                    for i, c in enumerate(s):
                        if c == '\u2013':
                            int_wrapped[j] = s[:i+1] + k + s[i+1:]
                    int_wrapped[j] = k + int_wrapped[j]
            else:
                int_wrapped = [k + s for s in v]
            long_list.extend(int_wrapped)
    r = ', '.join(long_list)

    # use "cluster X" as abbreviation when too long
    if len(r) > maxlen:
        global Wrap_N_c
        label_new = 'cluster {}'.format(Wrap_N_c)
        print('{} is: \n    {}'.format(label_new, r))
        Wrap_N_c += 1
        return r

    # add new lines for node shape
    return '\n'.join(__wrap_with_new_line([r], width))


def __wrap_with_new_line(lines, width):
    r = []
    for l in lines:

        # no space don't wrap
        if ' ' not in l:
            r.append(l)
        elif len(l) < width:
            r.append(l)
        else:
            # index of space which is closer to center => split there
            center = len(l)/2
            space_indices = [i for i, c in enumerate(l) if c == ' ']
            split_at = min(space_indices, key=lambda x: abs(x - center))
            l1 = l[:split_at]
            l2 = l[split_at+1:]  # drop the space

            # recursive wrapping until no space or short enough
            r.extend(__wrap_with_new_line([l1, l2], width))
    return r


def wraps(list1, maxlen=float('inf'), width=float('inf'), trimer=False):
    global Wrap_N_c
    Wrap_N_c = 0
    return list(map(lambda x: wrap_str(x, maxlen, width=width, trimer=trimer), list1))


# wrap an integer list
def int_wrap(long_list, to_str=False):
    if len(long_list) > 1:
        append_list = []
        long_list = sorted(map(int, long_list))
        x = long_list[0]  # start number of 'x-y'
        y = long_list[0]
        for n in long_list[1:]:
            if n != y + 1:
                if x + 1 < y:  # 1, 2 use 1 2 | 1, 2, 3 use 1-3
                    append_list.append(str(x) + '\u2013' + str(y))
                elif x == y - 1:
                    append_list.extend([str(x), str(y)])
                else:
                    append_list.append(str(x))
                x, y = n, n
            else:  # n == y + 1:
                y = n  # end number of 'x-y'
            if n == long_list[-1]:
                if x + 1 < y:  # 1, 2 use 1 2 | 1, 2, 3 use 1-3
                    append_list.append(str(x) + '\u2013' + str(y))
                elif x == y - 1:
                    append_list.extend([str(x), str(y)])
                else:
                    append_list.append(str(x))
    else:
        append_list = long_list
    if to_str:
        return ', '.join(append_list)
    else:
        return append_list
