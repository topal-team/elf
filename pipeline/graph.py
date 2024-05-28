from enum import Enum

class OperationType(Enum):
    RECV_FORWARD = 0
    FORWARD = 1
    SEND_FORWARD = 2
    RECV_BACKWARD = 3
    BACKWARD = 4
    SEND_BACKWARD = 5
    
    def __repr__(self) -> str:
        return self.name.lower()
    
    def __int__(self) -> int:
        return self.value

class Operation():
    def __init__(self, block_id, mb_id, op, rank, **options):
        self.block_id = block_id
        self.op = op
        self.mb_id = mb_id
        self.rank = rank
        self.options = options
        self.dependencies = []

    def add_dependency(self, node):
        self.dependencies.append(node)

    def __str__(self) -> str:
        return f'{self.block_id}:{repr(self.op)}({self.mb_id})'
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, __value: object) -> bool:
        return __value.op == self.op and __value.mb_id == self.mb_id and \
            __value.rank == self.rank and __value.options == self.options
    
    def __hash__(self) -> int:
        return hash((self.block_id, self.block_id, self.rank, self.op))

def print_graph(graph, level = 0):
    print("| " * level + str(graph))
    for d in graph.dependencies:
        print_graph(d, level + 1)

def graph_from_schedule(schedule):
    # For each one, add forward/backward dependencies (from the sequential nature)
    for operation in schedule:
        def match_op(op, block_id, op_type):
            return op.mb_id == operation.mb_id and op.block_id == block_id and op.op == op_type
        
        match operation.op:
            case OperationType.SEND_FORWARD:
                deps = [op for op in schedule if match_op(op, operation.block_id + 1, OperationType.RECV_FORWARD)]
            case OperationType.FORWARD:
                deps = [op for op in schedule if match_op(op, operation.block_id, OperationType.RECV_FORWARD) or 
                        match_op(op, operation.block_id - 1, OperationType.SEND_FORWARD)]
            case OperationType.SEND_BACKWARD:
                deps = [op for op in schedule if match_op(op, operation.block_id - 1, OperationType.RECV_BACKWARD)]
            case OperationType.BACKWARD:
                deps = [op for op in schedule if match_op(op, operation.block_id, OperationType.RECV_BACKWARD) or
                        match_op(op, operation.block_id + 1, OperationType.SEND_BACKWARD)]
            case _:
                deps = []
                
        for d in deps:
            operation.add_dependency(d)

    # Then, add the schedule dependencies from communications
    # A comm depends on the last comm on the same DEVICE, not BLOCK
    for i in range(len(schedule)):
        current_op = schedule[i]
        if current_op.op in [OperationType.FORWARD, OperationType.BACKWARD]:
            continue
        for j in reversed(range(i)):
            last_op = schedule[j]
            # Only send operations are blocking
            if last_op.op in [OperationType.RECV_FORWARD, OperationType.RECV_BACKWARD, OperationType.FORWARD, OperationType.BACKWARD] or \
                last_op.rank != current_op.rank:
                continue
            current_op.add_dependency(last_op)
            break

    last_ops = {}
    for operation in schedule:
        last_ops[(operation.rank, operation.mb_id)] = operation
    return last_ops # roots of the entire graph

def schedule_from_graph(graph):
    def dfs(node, visited, stack):
        # Mark the current node as visited
        if node in visited: return
        visited.add(node)
        
        # Recur for all the nodes dependent on this node
        for dependent in node.dependencies:
            if dependent not in visited:
                dfs(dependent, visited, stack)
        
        # Push current node to stack which stores the result
        stack.append(node)

    visited = set()
    stack = []
    
    # Call the recursive helper function to store Topological Sort
    # starting from all nodes one by one
    for root in graph.values():
        dfs(root, visited, stack)
    
    # Return contents of stack
    return stack

def fix_cycle(cycle):
    for op in cycle:
        if op.op not in [OperationType.FORWARD, OperationType.BACKWARD]:
            op.options["batch"] = True

def find_cycles(graph):
    def dfs(node, visited, stack, depth = 1, current_path = []):
        visited[node] = True
        stack[node] = depth  # To avoid cycles of length 2, we store the distance

        current_path.append(node)

        cycles = []
        for neighbor in node.dependencies:
            if neighbor not in visited:
                if cycle := dfs(neighbor, visited, stack, depth + 1, current_path):
                    cycles.extend(cycle)
            elif stack[neighbor] and stack[neighbor] < depth - 1:
                start_index = current_path.index(neighbor)
                cycle = current_path[start_index:]
                cycles.append(cycle)

        stack[node] = False
        current_path.pop()
        return cycles

    visited = {}
    stack = {}
    all_cycles = []
    for root in graph.values():
        cycles = dfs(root, visited, stack)
        all_cycles.extend(cycles)

    return all_cycles

def reorder_operations(operations):
    # Define the target and source operations
    target_ops = {OperationType.FORWARD, OperationType.BACKWARD}
    source_ops = {OperationType.RECV_FORWARD, OperationType.RECV_BACKWARD}
    
    # We need to iterate while keeping track of indexes because we'll modify the list
    i = 0
    while i < len(operations):
        current_op = operations[i]
        if current_op.op in target_ops:
            # Look for the next source operation
            for j in range(i + 1, len(operations)):
                next_op = operations[j]
                if next_op.op in source_ops:
                    # Move found operation to just before the current one
                    operations.insert(i, operations.pop(j))
                    i += 1
                    break
        i += 1
    return operations

