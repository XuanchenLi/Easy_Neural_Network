import json
import os
import datetime
import numpy as np
from ..core.node import *
from ..core.graph import *
from ..operators.loss import *
from ..operators.metrics import *


class Saver(object):
    def __init__(self, root_dir = ''):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

    def save(self, graph=None, meta=None, service_signature=None,
             model_file_name='model.json', weights_file_name='weights.npz'):
        if graph is None:
            graph = default_graph
        meta = {} if meta is None else meta
        meta['save_time'] = str(datetime.datetime.now())
        meta['weights_file_name'] = weights_file_name
        service = {} if service_signature is None else service_signature
        self.save_model_and_weights(
            graph, meta, service, model_file_name, weights_file_name
        )

    def save_model_and_weights(self, graph, meta, service,
                               model_file_name, weights_file_name):
        model_json = {
            'meta': meta,
            'service': service
        }
        graph_json = []
        weights_dict = dict()
        for node in graph.nodes:
            if not node.need_save:
                continue
            node.kargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
                'kargs': node.kargs
            }
            if node.value is not None:
                if isinstance(node.value, np.matrix):
                    node_json['dim'] = node.value.shape
            graph_json.append(node_json)
            if isinstance(node, Variable):
                weights_dict[node.name] = node.value

        model_json['graph'] = graph_json

        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4)
            print('Save model into file: {}'.format(model_file.name))
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'w') as weights_file:
            np.savez(weights_file, **weights_dict)
            print('Save weights to file: {}'.format(weights_file.name))

    def load(self, to_graph = None, model_file_name='model.json',
             weights_file_name = 'weights_npz'):
        if to_graph is None:
            to_graph = default_graph
        model_json = {}
        graph_json = []
        weights_dict = dict()
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz = np.load(weights_file)
            for file_name in weights_npz.fils:
                weights_dict[file_name] = weights_npz[file_name]
            weights_npz.close()

        graph_json = model_json['graph']
        self.reproduce_nodes(to_graph, graph_json, weights_dict)
        print('Load and restore model from {} and {}'.format(
            model_file_path, weights_file_path))

        self.meta = model_json.get('meta', None)
        self.service = model_json.get('service', None)
        return self.meta, self.service

    def reproduce_nodes(self, to_graph, graph_json, weights_json):
        for idx in range(len(graph_json)):
            node_json = graph_json[idx]
            node_name = node_json['name']
            weights = None
            if node_name in weights_json:
                weights = weights_json[node_name]
            target_node = get_node_from_graph(node_name, graph=to_graph)
            if target_node is None:
                target_node = Saver.create_node(
                    to_graph, graph_json, node_json
                )
            target_node.value = weights

    @staticmethod
    def create_node(graph, graph_json, node_json):
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dim = node_json.get('dim', None)
        kargs = node_json.get('kargs', None)
        kargs['graph'] = graph

        parents = []
