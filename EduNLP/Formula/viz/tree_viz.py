# coding: utf-8
# 2021/3/8 @ tongshiwei

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.tree._export import _MPLTreeExporter
from sklearn.tree._reingold_tilford import buchheim, Tree
from matplotlib.text import Annotation


class TreePlotter(_MPLTreeExporter):
    def recurse(self, node, ax, scale_x, scale_y, height, scale_x_offset=0, scale_y_offset=0, depth=0):
        kwargs = dict(bbox=self.bbox_args, ha='center', va='center',
                      zorder=100 - 10 * depth, xycoords='axes pixels')

        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + .5 + scale_x_offset) * scale_x, height - (node.y + .5 + scale_y_offset) * scale_y)

        if self.max_depth is None or depth <= self.max_depth:
            # if self.filled:
            #     kwargs['bbox']['fc'] = self.get_fill_color(tree,
            #                                                node.tree.node_id)
            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + .5 + scale_x_offset) * scale_x,
                             height - (node.parent.y + .5 + scale_y_offset) * scale_y)
                kwargs["arrowprops"] = self.arrow_args
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, ax, scale_x, scale_y, height,
                             scale_x_offset, scale_y_offset, depth=depth + 1)

        else:
            xy_parent = ((node.parent.x + .5) * scale_x,
                         height - (node.parent.y + .5) * scale_y)
            kwargs["arrowprops"] = self.arrow_args
            kwargs['bbox']['fc'] = 'grey'
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)

    def node_to_str(self, node: dict):
        return "\n".join(["%s: %s" % (k, v) for k, v in node.items()])

    def make_tree(self, ast: nx.DiGraph, root=None):
        if root is None:
            assert nx.is_tree(ast)
            for node in ast.nodes:
                if not list(ast.predecessors(node)):
                    root = node
                    break

        assert root is not None

        return self._make_tree(
            root, ast
        )

    def _make_tree(self, node, ast):
        return Tree(
            self.node_to_str(ast.nodes[node]),
            node,
            *[self._make_tree(successor, ast) for successor in ast.successors(node)]
        )

    def export(self, formula_ast, ax=None):
        self.filled = False

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        # my_tree = self._make_tree(0, decision_tree.tree_,
        #                           decision_tree.criterion)
        my_tree = self.make_tree(formula_ast)
        draw_tree = buchheim(my_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        self.recurse(draw_tree, ax,
                     scale_x, scale_y, ax_height)

        anns = [ann for ann in ax.get_children()
                if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [ann.get_bbox_patch().get_window_extent()
                       for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(scale_x / max_width,
                                                scale_y / max_height)
            for ann in anns:
                ann.set_fontsize(size)

        return anns


class ForestPlotter(TreePlotter):
    def export(self, forest, ax=None, root_list=None):
        self.filled = False

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        # my_tree = self._make_tree(0, decision_tree.tree_,
        #                           decision_tree.criterion)
        my_forest = []
        if root_list is None:
            if isinstance(forest, list):
                for tree in forest:
                    my_tree = self.make_tree(tree)
                    draw_tree = buchheim(my_tree)
                    my_forest.append(draw_tree)
        else:
            for tree in root_list:
                my_tree = self.make_tree(forest, tree)
                draw_tree = buchheim(my_tree)
                my_forest.append(draw_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        offset = np.stack([draw_tree.max_extents() + 1 for draw_tree in my_forest])
        max_x, max_y = sum(offset[:, 0]) + 1, max(offset[:, 1]) + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        x_offset = np.cumsum(offset[:, 0])
        x_offset = [0] + x_offset.tolist()
        for i, draw_tree in enumerate(my_forest):
            self.recurse(
                draw_tree, ax, scale_x, scale_y,
                ax_height,
                x_offset[i]
            )

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [ann.get_bbox_patch().get_window_extent()
                       for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(scale_x / max_width,
                                                scale_y / max_height)
            for ann in anns:
                ann.set_fontsize(size)

        return anns


#
# import networkx as nx
#
# g = nx.DiGraph()
# g.add_node(0, value=1, id=0)
# g.add_node(1, value=2, id=1)
# g.add_node(2, id=2)
# g.add_node(3, id=3)
# g.add_node(4, id=4)
# g.add_edge(0, 1)
# g.add_edge(0, 2)
# g.add_edge(1, 3)
# g.add_edge(3, 4)
#
# g2 = nx.DiGraph()
# g2.add_node(5, value=10, id=5)
# g2.add_node(6, value=200, id=6)
# g2.add_node(7, id=7)
# g2.add_edge(5, 6)
# g2.add_edge(5, 7)
# # ForestPlotter().export([g, g2])
# TreePlotter().export(g)
# plt.show()
