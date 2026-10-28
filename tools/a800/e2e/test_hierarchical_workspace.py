"""CPU regression for consuming Flux workspaces before a following segment reuses them.

Load only the actual HierarchicalRS class so this synchronization test does not
require importing Megatron, CUDA, or the frozen Flux extension on a login node.
"""
import ast
from pathlib import Path
import unittest


class Tensor:
    def __init__(self, value=0, rows=8):
        self.value = value
        self.shape = (rows, 1)

    def chunk(self, count, dim=0):
        return [Tensor(i + 1) for i in range(count)]

    def contiguous(self):
        return self

    def clone(self):
        return Tensor(self.value)

    def add_(self, other):
        self.value += other.value


class GuardedWorkspace:
    def __init__(self):
        self.consumers_pending = False

    def forward(self, tensor, weight, reduce_scatter_option=None):
        if self.consumers_pending:
            raise RuntimeError('a peer still reads the previous segment')
        self.consumers_pending = True
        return Tensor(tensor.value)

    def forward_barrier(self, tensor, weight):
        self.consumers_pending = False


class Request:
    def wait(self):
        pass


class Distributed:
    isend = 'send'
    irecv = 'receive'

    def P2POp(self, kind, tensor, rank, group):
        if kind == self.irecv:
            tensor.value = 10
        return kind

    def batch_isend_irecv(self, operations):
        return [Request() for _ in operations]


class Torch:
    distributed = Distributed()

    @staticmethod
    def empty_like(tensor):
        return Tensor()


class WorkspaceRegression(unittest.TestCase):
    def test_segments_and_repeated_calls_finish_consumers(self):
        source = Path(__file__).parent/'distributed/adapter.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HierarchicalRS')
        namespace = {'torch': Torch()}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        for nodes in (2, 3, 4):
            for node in range(nodes):
                with self.subTest(nodes=nodes, node=node):
                    op = namespace['HierarchicalRS'].__new__(namespace['HierarchicalRS'])
                    op.nodes, op.node, op.local_tp = nodes, node, 4
                    op.rank, op.world, op.base, op.group = node*4, nodes*4, 0, None
                    op.local_op = GuardedWorkspace()
                    for _ in range(2):
                        result = op.forward(Tensor(rows=nodes*4), None)
                        self.assertEqual(result.value, node + 1 + 10*(nodes - 1))
                        self.assertFalse(op.local_op.consumers_pending)


if __name__ == '__main__':
    unittest.main()
