from .node import Variable
from .graph import default_graph


class NameScope(object):
    def __init__(self, name_scope):
        self.name_scope = name_scope

    def __enter__(self):
        default_graph.name_scope = self.name_scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        default_graph.name_scope = None


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + "/" + node_name
    for node in graph.nodes:
        if node_name == node.name:
            return node
    return None


def update_node_value(node_name, new_value, name_scope=None, graph=None):
    node = get_node_from_graph(node_name, name_scope, graph)
    if node is not None and new_value.shape == node.value.shape:
        node.value = new_value


def get_trainable_variables(node_name = None, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in graph.nodes if isinstance(node, Variable) and node.trainable]

    if name_scope:
        node_name = name_scope + '/' +node_name
    return get_node_from_graph(node_name, graph=graph)
