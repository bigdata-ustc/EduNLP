# coding: utf-8
# 2021/3/8 @ tongshiwei

import matplotlib.pyplot as plt
from sklearn.tree._export import _MPLTreeExporter
from sklearn.tree._reingold_tilford import buchheim, Tree
from matplotlib.text import Annotation


class TreePlotter(_MPLTreeExporter):
    def recurse(self, node, ax, scale_x, scale_y, height, depth=0):
        kwargs = dict(bbox=self.bbox_args, ha='center', va='center',
                      zorder=100 - 10 * depth, xycoords='axes pixels')

        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + .5) * scale_x, height - (node.y + .5) * scale_y)

        if self.max_depth is None or depth <= self.max_depth:
            # if self.filled:
            #     kwargs['bbox']['fc'] = self.get_fill_color(tree,
            #                                                node.tree.node_id)
            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + .5) * scale_x,
                             height - (node.parent.y + .5) * scale_y)
                kwargs["arrowprops"] = self.arrow_args
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, ax, scale_x, scale_y, height,
                             depth=depth + 1)

        else:
            xy_parent = ((node.parent.x + .5) * scale_x,
                         height - (node.parent.y + .5) * scale_y)
            kwargs["arrowprops"] = self.arrow_args
            kwargs['bbox']['fc'] = 'grey'
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)

    def _make_forest(self, ast):
        forest = []
        for node in ast:
            if node["structure"]["father"] is None:
                return Tree()
            else:
                pass

        return Tree(name, node_id, *children)

    def export(self, formula_ast, ax=None):
        self.filled = False

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        # my_tree = self._make_tree(0, decision_tree.tree_,
        #                           decision_tree.criterion)
        my_tree = self._make_forest(formula_ast)
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
