from .node import*


class Graph(object):
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)

    def clear_jacobi(self):
        for node in self.nodes:
            node.clear_jacobi()

    def clear_value(self):
        for node in self.nodes:
            node.clear_value(False)

    def node_num(self):
        return len(self.nodes)


default_graph = Graph()
