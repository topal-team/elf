from elf.scheduling.comm_scheduling import schedule_to_graph


def simulate(schedule, profiles):
	graph = schedule_to_graph(
		schedule
	)  # turns out comm scheduling representation works without communications as well

	timeline = {}

	def _duration(node):
		# duration of op schedule[node]
		op = schedule[node]
		duration = profiles[op.block_id].duration(op)
		return duration

	def _transfer(a, b):
		# transfer time between ops a and b
		if schedule[a].rank != schedule[b].rank:
			return profiles[schedule[a].block_id].t_transfer
		return 0.0

	def _compute_times(node):
		if schedule[node] in timeline:
			return timeline[schedule[node]]

		start = max(
			(_compute_times(succ)[1] + _transfer(succ, node) for succ in graph.get_successors(node)),
			default=0,
		)
		timeline[schedule[node]] = (start, start + _duration(node))
		return start

	for node in schedule:
		_compute_times(node)  # fill timeline

	return timeline
