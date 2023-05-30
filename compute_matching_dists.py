import cogent3
from corruption_experiment import compare_distance
from tree import Tree


def str_to_tuple(s, start=0, stop=None):
    out = []
    if stop is None:
        stop = len(s)

    skip_to = 0
    part = ""
    for i in range(start, stop):
        if i < skip_to:
            continue

        c = s[i]
        # print(i, c)
        if c == " ":
            continue
        elif c == "(":
            part, skip_to = str_to_tuple(s, start=i + 1, stop=stop)
        elif c == ")":
            out.append(part)
            return tuple(out), i + 1
        elif c == ",":
            out.append(part)
            part = ""
        else:
            part += c
    return tuple(part)


def main():
    print("Loading ZT from File")
    zt = cogent3.load_tree("data/zhu-tree-rooted-resolved-molclock.nwk")

    with open("dcm_simulated.txt", "r") as f:
        for line in f:
            a, b = line.strip().split(";")[-2:]
            # with open("comput.txt", "r") as f:
            #     a = f.read().strip()
            # print(a)
            print("Converting to tuple")
            ans = str_to_tuple(a)
            bns = str_to_tuple(b)
            # print(ans)
            print("Converting to Tree")
            atree = Tree.construct_from_tuple(ans)
            btree = Tree.construct_from_tuple(bns)

            st = zt.get_sub_tree(atree.get_descendants())

            print("Computing Matching Distance", len(atree))
            a = compare_distance(cogent3.make_tree(str(atree)), st)
            b = compare_distance(cogent3.make_tree(str(btree)), st)

            with open("dcm_simulated_with_distances.txt", "a") as g:
                g.write(line.strip())
                g.write(";")
                g.write(str(a))
                g.write(";")
                g.write(str(b))
                g.write("\n")


if __name__ == "__main__":
    main()
