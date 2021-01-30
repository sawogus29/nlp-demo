from graphviz import Digraph

def draw_tree(root, get_children, get_print, get_label):
    queue = list()
    prev_depth = 0
    dot = Digraph()
    queue.append((root, -1))
    
    id = 0
    while len(queue) != 0:
        current, parent_id = queue.pop(0)
        dot.node(str(id), get_print(current))
        if parent_id != -1:
            dot.edge(str(parent_id), str(id), label=get_label(current))

        for child in get_children(current):
            queue.append((child, id))

        id += 1
    
    return dot.pipe('svg').decode('utf-8')

def draw_dep_tree(span):
    # i = -1

    # find ROOT
    # for token in span:
        # if token.dep_ == 'ROOT':
            # i = token.i - span[0].i
            # break

    # if i != -1:
    return draw_tree(span.root, 
        lambda current: current.children,
        lambda current: current.text,
        lambda current: current.dep_)

def preprocess(input_str):
    out = input_str
    out = out.replace('“','"')
    out = out.replace('”','"')
    ps = out.splitlines()
    return filter(lambda x: True if len(x) > 0 else False, ps)
