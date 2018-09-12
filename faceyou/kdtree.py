from collections import namedtuple
from operator import itemgetter
from pprint import pformat


# 定义Node类，
class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        # 返回格式化数组，便于读取
        return pformat(tuple(self))


# 创建kd树
def kdtree(point_list, depth=0):
    try:
        k = len(point_list[0])  # 假设所有节点具有所有相同维度
    except IndexError as e:  # if not point_list:
        return None
    # 根据深度确定选取的维度
    # 由于是取余，则取值在[0,k-1]　k表示维度
    axis = depth % k

    # 根据所选定维度对节点列表排序
    # 这里sort(key＝xxx)中的xxx是一个可迭代对象，指定元素来排列，这里是根据第axis个元素排列
    # 我刚才尝试的想法是operator.itemgetter，但是返回的是operator类，而不是itemgetter迭代器，所以报错
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2  # choose median

    # 创建节点并构建子树
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )


def main():
    """Example usage"""
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = kdtree(point_list)
    print(tree)


if __name__ == '__main__':
    main()
