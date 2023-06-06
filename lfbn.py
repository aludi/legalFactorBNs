import random
import re
from anytree import Node, RenderTree, PreOrderIter, PostOrderIter
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gumimage
import itertools



#from IPython.display import Image # to display the exported images


def make_tree(a, t):

    if len(t.leaves) > 1:
        return t
    else:
        if random.random() < 0.3:
            t1 = Node("NOT", val=None)
            t.parent = t1
            return make_tree(a, t1)
        else:
            # pick new item
            a2 = random.choice(a)
            a.remove(a2)
            t1 = Node(random.choice(["AND", "OR"]), val=None)
            # random choice left or right branching
            if random.random() >= 0.5:
                tx = Node(a2, parent=t1, val=None)
                t.parent = t1
            else:
                t.parent = t1
                tx = Node(a2, parent=t1, val=None)

            return make_tree(a, t1)


def three_val_log(p, q, op):
    if p in [0, 1] and q in [0, 1]:
        if op == "AND":
            return int(p and q)
        elif op == "OR":
            return int(p or q)
    else:
        if op == "OR":
            return max(p, q)
        elif op == "AND":
            return max(min(p, q), 0)    # return 0 if false


def eval_tree(t):
    #### three-valued logic?????? my bro???
    if t.val == 1 or t.val == 0 or t.val == -1:
        return t.val

    if len(t.children) == 1:
        left_val = eval_tree(t.children[0])
        right_val = None
    else:
        left_val = eval_tree(t.children[0])
        right_val = eval_tree(t.children[1])

    if t.name == "AND":
        t.val = three_val_log(left_val, right_val, "AND")
        return t.val
        #t.val = int(left_val and right_val)
        #return int(left_val and right_val)
    if t.name == "OR":
        t.val = three_val_log(left_val, right_val, "OR")
        return t.val
        #t.val = int(left_val or right_val)
        #return int(left_val or right_val)
    if t.name == "NOT":
        if left_val == 1:
            t.val = 0
        elif left_val == 0:
            t.val = 1
        else:
            t.val = -1
        return t.val


def get_leaf_node_vals(t, valuation):
    node_vals = []
    for leaf_node in t.leaves:
        node_vals.append((leaf_node.name, valuation[leaf_node.name]))
    return node_vals

def get_leaf_nodes(t):
    node_vals = []
    for leaf_node in t.leaves:
        node_vals.append((leaf_node.name))
    return node_vals



def evaluate_tree(t, valuation):
    # set valuation on the leaf nodes

    for n in PostOrderIter(t):
        n.val = None

    for leaf_node in t.leaves:
        leaf_node.val = valuation[leaf_node.name]
    for pre,fill, node in RenderTree(t):
        print("%s%s%s"%(pre, node.name, node.val))
    prob = eval_tree(t)
    for pre,fill, node in RenderTree(t):
        print("%s%s%s"%(pre, node.name, node.val))

    return prob

def evaluate_tree_set(S, valuation):
    # do forward chaining



    for lhs, rhs in S:
        evaluate_tree(rhs, valuation)
        if valuation[lhs] != rhs.root.val:
            if valuation[lhs] == -1:
                valuation[lhs] = rhs.root.val
                print(f"updated value {lhs} to {rhs.root.val}")
            else:
                print("conflict")
                print(lhs, rhs.root.val, valuation[lhs])

    for x in valuation.keys():
        print(x, valuation[x])
    return valuation
    #exit()

def check_acyclic(S):
    for (lhs, tree) in S:
        n = tree.leaves()
    pass

def create_dag(S):
    bn = gum.BayesNet('logicNet')
    nodes = []
    dict_v = {}
    for (lhs, tree) in S:
        dict_v[lhs] = tree
        if lhs not in nodes:
            nodes.append(lhs)
        for n in tree.leaves:
            if n.name not in nodes:
                nodes.append(n.name)
    print(nodes)


    BN_nodes = []
    for node in nodes:
        x = gum.LabelizedVariable(node, node, 2)
        y = bn.add(x)
        print(x)
        BN_nodes.append((x, y)) # add node and node id
    print(BN_nodes)


    for (n, i) in BN_nodes:
        print(n)
        for (lhs, tree) in S:
            print(n, n.name(), lhs)
            if n.name() == lhs:

                for q in tree.leaves:
                    print(q.name, n.name())
                    try:
                        bn.addArc(q.name, n.name())
                    except gum.InvalidDirectedCycle:
                        print("added directed cycle - error")
                        exit("exit")
                        return 0

    print(bn)

    for (n, c) in BN_nodes:
        print(n.name(), c)
        if n.name() in dict_v:
            print(f"{n.name()} is on LHS")
            tree = dict_v[n.name()]
        else:
            print(f"{n.name()} is not on LHS")
            tree = Node(n.name())
        print("tree", tree)
        c_a = bn.cpt(c)
        #print("nodes dict? ", bn.cpt(i))
        i=gum.Instantiation(c_a)
        i.setFirst()
        s = 0.0
        while (not i.end()):

            print(i)
            d = i.todict()
            print("set", c_a[i])
            print(tree),
            print(i.todict())
            x = evaluate_tree(tree, i.todict())
            print(f"new value of {n.name()} {c_a[i]}", x)
            print("value of i", i)
            print("name", n.name())
            if d[n.name()] == 0:
                c_a[i] = 1 - x
            else:
                c_a[i] = x

            s += c_a.get(i)
            i.inc()
            print(s)
        print(bn.cpt(c))

        #print(I)


    gum.saveBN(bn, 'bnsave.net')

    gumimage.export(bn, "test_export.png")  # a causal model has a toDot method.
    gumimage.exportInference(bn, "test_export1.pdf")

def traverse_inorder(r, s):
    #print("String   ", s)
    if len(r.children) == 1:
        s1 = r.name
        s1 += traverse_inorder(r.children[0], s1)
        return f"{s1}"
    elif len(r.children) == 0:
        s = r.name
        return s
    else:
        s = traverse_inorder(r.children[0], s)
        s = "("+s
        s += r.name

        if len(r.children) > 1:
            s1 = traverse_inorder(r.children[1], s)
            s += f"{s1}"
        s += ")"

    return s

def get_combinations(priors):   # ids
    #print("PRIOR LIST")
    l = list(itertools.product([1, 0], repeat=len(priors)))
    #print(l)
    dict_list = []
    for val in l:
        d = {}
        for i in range(0, len(priors)):
            d[priors[i]] = val[i]
        dict_list.append(d)
    return dict_list


def predict_output(S):
    bn = gum.loadBN("bnsave.net")
    prior_nodes = []
    output_node = []
    for name in bn.names():
        node_id = bn.idFromName(name)
        if len(bn.parents(node_id)) == 0:
            prior_nodes.append(node_id)
        if len(bn.children(node_id)) == 0:
            output_node.append(node_id)

    val_dict = {}
    know_val_dict = {}

    dict_list_valuations = get_combinations(prior_nodes)
    predicted_valuations = []

    for valuation in dict_list_valuations:
        print(val_dict)
        for node_name in bn.names():
            if bn.idFromName(node_name) in valuation.keys():
                val_dict[node_name] = valuation[bn.idFromName(node_name)]
            else:
                val_dict[node_name] = -1

        ie = gum.LazyPropagation(bn)
        ie.setEvidence(valuation)
        ie.makeInference()
        predicted_valuations.append(f"for valuation {val_dict}:")
        p_dict_val = evaluate_tree_set(S, val_dict)
        predicted_valuations.append(f"for inferred valuation {p_dict_val}:")


        #for lhs, rhs in S:
        #p = evaluat
        #
        # e_tree(rhs, val_dict)
        # BREAKs
        predicted_valuations.append(f"v({lhs}) == {p_dict_val[lhs]}")
        #print(ie.posterior(lhs))
        post = ie.posterior(lhs)[1]

        predicted_valuations.append(f"P({lhs}) == {post}")
        predicted_valuations.append({int(post) == p_dict_val[lhs]})
        predicted_valuations.append("\n\n\n")


    for x in predicted_valuations:
        print(x)


    #gumimage.exportInference(bn, "test_export_inference.png", evs=val_dict)


    #print(val_dict)


    # evaluate the network for all
    # tree inference
    # compare with truth table output

def rewrite_rules(S, x, list_t):
    print("rules in S:")
    for l, r in S:
        print(l, r.leaves)

    #exit()

def rewrite_rules2(rule_set, S, list_t):
    if len(rule_set) == 0:
        print("new reduced rules")
        for rule in list_t:
            for pre, fill, node in RenderTree(rule[1]):
                print("%s%s %s" % (pre, node.name, node.val))
        print(list_t)
        return list_t
    n = rule_set[0]
    n_l, n_r = n
    flag = 0
    for lh, t in list_t:
        for node1 in PostOrderIter(t):
            print(node1.name, n_l)
            if node1.name == n_l:
                n_r.parent = node1.parent
                node1.parent=None
                flag = 1
    for node1 in PostOrderIter(n_r):
        for (lh, t) in list_t:
            print(node1.name, lh)
            if lh == node1.name:
                t.parent = node1.parent
                node1.parent = None
                flag = 1

    if flag == 0:  # rule does not fit with tree
        list_t.append((n_l, n_r))

    rewrite_rules(rule_set[1:], list_t)





def rewrite_rules_1(S):   # rewrite rewrite algorithm
    l_l = []
    rule_replaced = []

    for i in range(0, len(S)):
        (lhs, rhs) = S[i]
        for (r_l, r_r) in S[i+1:]:
            for node1 in PostOrderIter(r_r):
                if node1.name == lhs:
                    print("merge needed", node1)
                    rhs.parent = node1
                    l_l.append(r_l)


    l = []
    for (lhs, rhs) in S:
        if rhs.root not in l:
            l.append(rhs.root)
            #l_l.append(lhs) # todo bug

    print("printing unique root trees")
    for rhs in l:
        print(rhs)
        for pre, fill, node in RenderTree(rhs):
            print("%s%s %s" % (pre, node.name, node.val))

    return zip(l_l, l) # consolidate the sentences into one big sentence -> equivalence





def save_rules(S):
    string_t = "RULES \n\n\n"
    for l, r in S:
        #print("\nTRAVERSE INORDER \n")
        #for pre, fill, node in RenderTree(r):
        #    print("%s%s%s" % (pre, node.name, node.val))
        s = traverse_inorder(r, "")
        #print("end ", s)
        string_t = string_t + f"{l} <=> {s} \n\n"
    textfile = open("ruleset.txt", "w")
    textfile.write(string_t)
    textfile.close()

def save_rules_eq(S):
    string_t = "appended rules \n\n\n"
    for l, r in S:
        #print("\nTRAVERSE INORDER \n")
        #for pre, fill, node in RenderTree(r):
        #    print("%s%s%s" % (pre, node.name, node.val))
        s = traverse_inorder(r, "")
        #print("end ", s)
        string_t = string_t + f"{l} <=> {s} \n\n"
    textfile = open("ruleset.txt", "a")
    textfile.write(string_t)
    textfile.close()



def gen_sentence(a, s):
    if len(s) > max_sen_length:
        return s
    else:
        a = [a1 for a1 in a if a1 not in s]

        if random.random() < 0.5:
            s1 = f"(NOT({s}))"
            return gen_sentence(a, s1)
        x = random.choice(a)
        if random.random() < 0.5:
            s1 = f"({x} AND {s})"
        else:
            s1 = f"({x} OR {s})"
        return gen_sentence(a, s1)



atoms = ["a", "b", "c", "d", "e", "f", "g", "h"]
l_atoms = len(atoms)
connectives = ["AND", "OR", "NOT"]   # conjunction, disjunction, negation

max_sen_length = 7
LHS = []
num_S = 4    # num_S < l_atoms
S = []
rhs_all = []


for i in range(0, num_S):
    lhs = random.choice(atoms)
    while lhs in LHS or lhs in rhs_all:
        lhs = random.choice(atoms)
    LHS.append(lhs)

    valuation = {'a':0, 'b':0,
                 'c':0, 'd':0,
                 'e':0, 'f':0,
                 'g':0, 'h':0}


    a = [a for a in atoms if a not in lhs]
    s = random.choice(a)
    a.remove(s)

    rhs = make_tree(a, Node(s))


    l = get_leaf_nodes(rhs)
    for node in l:
        rhs_all.append(node)


    #for pre,fill, node in RenderTree(rhs):
    #    print("%s%s"%(pre, node.name))
    #rhs = gen_sentence(a, random.choice(a))
    #print(f"{lhs} <=> {rhs}")
    evaluate_tree(rhs, valuation)
    #for pre,fill, node in RenderTree(rhs):
    #    print("%s%s %s"%(pre, node.name, node.val))

    S.append((lhs, rhs))


create_dag(S)
save_rules(S)
l = rewrite_rules(S, [], [])
#save_rules_eq(l)
predict_output(S)


