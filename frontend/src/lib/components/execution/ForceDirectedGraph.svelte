<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import * as d3 from 'd3';
	import type { ExecutionGraph, ExecutionNode, ExecutionStatus } from '$lib/types/api';

	interface Props {
		graph: ExecutionGraph;
	}

	let { graph }: Props = $props();

	let svgContainer = null as unknown as HTMLDivElement;
	let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
	let g: d3.Selection<SVGGElement, unknown, null, undefined>;
	let simulation: d3.Simulation<GraphNode, undefined> | null = null;
	let tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>;

	interface GraphNode extends d3.SimulationNodeDatum {
		id: string;
		node: ExecutionNode;
	}

	interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
		source: GraphNode;
		target: GraphNode;
	}

	// Status color mapping
	const statusColors: Record<ExecutionStatus, string> = {
		RUNNING: '#3b82f6',    // Blue
		COMPLETED: '#10b981',  // Green
		FAILED: '#ef4444'      // Red
	};

	// Node type icon mapping
	const nodeTypeIcons: Record<string, string> = {
		TOOL: '🔧',
		AGENT: '🤖',
		WORKFLOW_STAGE: '📊'
	};

	onMount(() => {
		initChart();
	});

	onDestroy(() => {
		if (simulation) {
			simulation.stop();
		}
		if (tooltip) {
			tooltip.remove();
		}
	});

	function initChart() {
		const width = svgContainer.clientWidth;
		const height = svgContainer.clientHeight;

		// Validate non-zero dimensions before D3 init
		if (width === 0 || height === 0) {
			console.warn('Container has zero dimensions, deferring chart initialization');
			return;
		}

		// Create SVG
		svg = d3.select(svgContainer)
			.append('svg')
			.attr('width', '100%')
			.attr('height', '100%')
			.attr('viewBox', `0 0 ${width} ${height}`);

		// Create tooltip
		tooltip = d3.select('body')
			.append('div')
			.style('position', 'absolute')
			.style('visibility', 'hidden')
			.style('background-color', 'white')
			.style('border', '1px solid #e5e7eb')
			.style('border-radius', '6px')
			.style('padding', '8px')
			.style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
			.style('font-size', '12px')
			.style('pointer-events', 'none')
			.style('z-index', '1000');

		// Create group for zoom/pan
		g = svg.append('g');

		// Setup zoom
		const zoom = d3.zoom<SVGSVGElement, unknown>()
			.scaleExtent([0.1, 4])
			.on('zoom', (event) => {
				g.attr('transform', event.transform);
			});

		svg.call(zoom);

		// Initial render
		updateChart();
	}

	function updateChart() {
		if (!svg || !g) return;

		const width = svgContainer.clientWidth;
		const height = svgContainer.clientHeight;

		// Convert nodes to D3 format
		const nodes: GraphNode[] = Object.values(graph.nodes).map(node => ({
			id: node.node_id,
			node
		}));

		// Create links from parent relationships
		const links: GraphLink[] = [];
		Object.values(graph.nodes).forEach(node => {
			if (node.parent_id) {
				const source = nodes.find(n => n.id === node.parent_id);
				const target = nodes.find(n => n.id === node.node_id);
				if (source && target) {
					links.push({ source, target });
				}
			}
		});

		// Stop existing simulation
		if (simulation) {
			simulation.stop();
		}

		// Create force simulation
		simulation = d3.forceSimulation(nodes)
			.force('link', d3.forceLink<GraphNode, GraphLink>(links)
				.id(d => d.id)
				.distance(100))
			.force('charge', d3.forceManyBody().strength(-300))
			.force('collision', d3.forceCollide().radius(30))
			.force('center', d3.forceCenter(width / 2, height / 2))
			.force('x', d3.forceX(width / 2).strength(0.1))
			.force('y', d3.forceY(height / 2).strength(0.1));

		// Clear existing elements
		g.selectAll('*').remove();

		// Create links
		const link = g.append('g')
			.selectAll('line')
			.data(links)
			.join('line')
			.attr('stroke', '#d1d5db')
			.attr('stroke-width', 2)
			.attr('stroke-opacity', 0.6);

		// Create nodes
		const node = g.append('g')
			.selectAll('g')
			.data(nodes)
			.join('g')
			.call(d3.drag<SVGGElement, GraphNode>()
				.on('start', dragStarted)
				.on('drag', dragged)
				.on('end', dragEnded) as any);

		// Add circles
		node.append('circle')
			.attr('r', 20)
			.attr('fill', d => statusColors[d.node.status])
			.attr('stroke', '#fff')
			.attr('stroke-width', 2);

		// Add icons (text emoji)
		node.append('text')
			.attr('text-anchor', 'middle')
			.attr('dominant-baseline', 'middle')
			.attr('font-size', '16px')
			.text(d => nodeTypeIcons[d.node.node_type] || '?');

		// Add labels below nodes
		node.append('text')
			.attr('text-anchor', 'middle')
			.attr('dy', 30)
			.attr('font-size', '11px')
			.attr('fill', '#374151')
			.text(d => d.node.name.length > 20 ? d.node.name.substring(0, 17) + '...' : d.node.name);

		// Add tooltip on hover
		node.on('mouseenter', (event, d) => {
			const duration = d.node.duration_ms !== null
				? `${(d.node.duration_ms / 1000).toFixed(2)}s`
				: 'Running...';

			let tooltipContent = `
				<div style="font-weight: 600; margin-bottom: 4px;">${d.node.name}</div>
				<div style="color: #6b7280; margin-bottom: 2px;">Type: ${d.node.node_type}</div>
				<div style="color: #6b7280; margin-bottom: 2px;">Status: ${d.node.status}</div>
				<div style="color: #6b7280; margin-bottom: 2px;">Duration: ${duration}</div>
			`;

			if (d.node.error) {
				tooltipContent += `<div style="color: #ef4444; margin-top: 4px;">Error: ${d.node.error}</div>`;
			}

			tooltip
				.html(tooltipContent)
				.style('visibility', 'visible')
				.style('top', `${event.pageY - 10}px`)
				.style('left', `${event.pageX + 10}px`);
		})
		.on('mousemove', (event) => {
			tooltip
				.style('top', `${event.pageY - 10}px`)
				.style('left', `${event.pageX + 10}px`);
		})
		.on('mouseleave', () => {
			tooltip.style('visibility', 'hidden');
		});

		// Update positions on simulation tick
		if (simulation) {
			simulation.on('tick', () => {
				link
					.attr('x1', d => (d.source as GraphNode).x || 0)
					.attr('y1', d => (d.source as GraphNode).y || 0)
					.attr('x2', d => (d.target as GraphNode).x || 0)
					.attr('y2', d => (d.target as GraphNode).y || 0);

				node.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`);
			});
		}
	}

	function dragStarted(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
		if (!event.active && simulation) simulation.alphaTarget(0.3).restart();
		event.subject.fx = event.subject.x;
		event.subject.fy = event.subject.y;
	}

	function dragged(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
		event.subject.fx = event.x;
		event.subject.fy = event.y;
	}

	function dragEnded(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
		if (!event.active && simulation) simulation.alphaTarget(0);
		event.subject.fx = null;
		event.subject.fy = null;
	}

	// React to graph changes
	$effect(() => {
		if (svg && graph) {
			updateChart();
		}
	});
</script>

<div bind:this={svgContainer} style="width: 100%; height: 600px; border: 1px solid #e5e7eb; border-radius: 8px; background: #f9fafb;"></div>
