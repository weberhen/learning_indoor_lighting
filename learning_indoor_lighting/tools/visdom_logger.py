# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Labal
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from visdom import Visdom
import numpy as np
import numbers


class VisdomLogger:
    """
    The visualization class provides an easy access to some of the visdom functionalities
    Accept as input a number that will be ploted over time or an image of type np.ndarray
    """
    items_iterator = {}
    items_to_visualize = {}
    windows = {}
    vis = Visdom()

    def check_availability(vis):
        # check if the Visdom server is running. only once.
        is_done = vis.text('visdom check')
        if is_done is False:
            raise RuntimeError('Visdom server is not running. Run the server first: python -m visdom.server')
        else:
            print('Visdom available at: %s:%s' % (vis.server, vis.port))
            vis.close()  # close visdom check
    check_availability(vis)

    @classmethod
    def visualize(cls, item, name, **args):
        """
        Visualize an item in a new window (if the parameter "name" is not on the list of previously given names) or
        updates an existing window identified by "name"
        :param item:   Item to be visualized (a number or a numpy image).
        :param name:   String to identify the item.
        :param args:  dict containing options for visdom
        """
        if name not in cls.items_to_visualize:
            cls.new_item(item, name, **args)
        else:
            cls.update_item(item, name, **args)
        cls.items_to_visualize[name] = item

    @classmethod
    def new_item(cls, item, name, **args):
        if isinstance(item, numbers.Number):
            cls.items_iterator[name] = 0
            win = cls.vis.line(
                X=np.array([cls.items_iterator[name]]),
                Y=np.array([item]),
                opts=dict(title=name)
            )
            cls.windows[name] = win
        elif isinstance(item, np.ndarray):
            win = cls.vis.image(
                item,
                opts=args,
            )
            cls.windows[name] = win
        else:
            print("type {} not supported for visualization".format(type(item)))

    @classmethod
    def update_item(cls, item, name, **args):
        if isinstance(item, numbers.Number):
            cls.vis.line(
                # to plot the number we need to give its position in the x axis hence we keep track of how many times we
                # updates this item (stored in items_iterator)
                X=np.array([cls.items_iterator[name]]),
                Y=np.array([item]),
                win=cls.windows[name],
            )
            cls.items_iterator[name] += 1
        elif isinstance(item, np.ndarray):
            cls.vis.image(
                item,
                opts=args,
                win=cls.windows[name]
            )
        else:
            print("type {} not supported for visualization".format(type(item)))
