/**
 * Execution tracking state management with WebSocket integration
 */

import { writable, derived } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

// WebSocket URL for events
const WS_URL = import.meta.env.VITE_WS_URL ||
	(typeof window === 'undefined' ? 'ws://daemon:8484' : 'ws://localhost:8484');

interface ExecutionStoreState {
	activeGraphs: Record<string, T.ExecutionGraph>;
	selectedWorkflowId: string | null;
	wsConnected: boolean;
	error: string | null;
}

// Create execution store
function createExecutionStore() {
	const { subscribe, update } = writable<ExecutionStoreState>({
		activeGraphs: {},
		selectedWorkflowId: null,
		wsConnected: false,
		error: null
	});

	let ws: WebSocket | null = null;
	let reconnectTimeout: number | null = null;
	let reconnectAttempts = 0;
	const MAX_RECONNECT_ATTEMPTS = 5;
	const RECONNECT_DELAY = 3000;

	// Connect to WebSocket
	function connect() {
		if (typeof window === 'undefined') return; // Skip SSR

		try {
			ws = new WebSocket(`${WS_URL}/ws/events`);

			ws.onopen = () => {
				console.log('WebSocket connected');
				reconnectAttempts = 0;
				update(state => ({ ...state, wsConnected: true, error: null }));
			};

			ws.onmessage = (event) => {
				try {
					const data: T.DashboardEvent = JSON.parse(event.data);

					// Ignore ping messages
					if ((data as any).type === 'ping') return;

					// Handle execution node events
					if (data.event_type === 'EXECUTION_NODE_START' || data.event_type === 'EXECUTION_NODE_COMPLETE') {
						const { workflow_id, node } = data.data as {
							workflow_id: string;
							node: T.ExecutionNode;
						};

						update(state => {
							const graph = state.activeGraphs[workflow_id];
							if (!graph) {
								// Create new graph if doesn't exist
								const newGraph: T.ExecutionGraph = {
									workflow_id,
									symbol: node.metadata?.symbol as string | null || null,
									root_node_id: node.parent_id === null ? node.node_id : null,
									nodes: { [node.node_id]: node },
									created_at: new Date().toISOString(),
									updated_at: new Date().toISOString()
								};
								return {
									...state,
									activeGraphs: { ...state.activeGraphs, [workflow_id]: newGraph },
									selectedWorkflowId: state.selectedWorkflowId || workflow_id
								};
							}

							// Update existing graph
							const updatedGraph = {
								...graph,
								nodes: { ...graph.nodes, [node.node_id]: node },
								updated_at: new Date().toISOString()
							};

							// Set root if not set and node has no parent
							if (!updatedGraph.root_node_id && node.parent_id === null) {
								updatedGraph.root_node_id = node.node_id;
							}

							// Remove from activeGraphs if all nodes completed/failed
							const allDone = Object.values(updatedGraph.nodes).every(
								n => n.status === 'COMPLETED' || n.status === 'FAILED'
							);

							if (allDone) {
								const { [workflow_id]: _, ...remainingGraphs } = state.activeGraphs;
								return {
									...state,
									activeGraphs: remainingGraphs,
									selectedWorkflowId: state.selectedWorkflowId === workflow_id
										? Object.keys(remainingGraphs)[0] || null
										: state.selectedWorkflowId
								};
							}

							return {
								...state,
								activeGraphs: { ...state.activeGraphs, [workflow_id]: updatedGraph }
							};
						});
					}
				} catch (error) {
					console.error('Failed to parse WebSocket message:', error);
				}
			};

			ws.onerror = (error) => {
				console.error('WebSocket error:', error);
				update(state => ({ ...state, error: 'WebSocket connection error' }));
			};

			ws.onclose = () => {
				console.log('WebSocket disconnected');
				update(state => ({ ...state, wsConnected: false }));

				// Attempt reconnect
				if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
					reconnectAttempts++;
					console.log(`Reconnecting (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
					reconnectTimeout = window.setTimeout(connect, RECONNECT_DELAY);
				} else {
					update(state => ({
						...state,
						error: 'Failed to connect to WebSocket after multiple attempts'
					}));
				}
			};
		} catch (error) {
			console.error('Failed to create WebSocket:', error);
			update(state => ({ ...state, error: 'Failed to create WebSocket connection' }));
		}
	}

	// Disconnect from WebSocket
	function disconnect() {
		if (reconnectTimeout !== null) {
			clearTimeout(reconnectTimeout);
			reconnectTimeout = null;
		}
		if (ws) {
			ws.close();
			ws = null;
		}
	}

	// Fetch active graphs from API
	async function fetchActive() {
		try {
			const response = await api.getActiveExecutionGraphs();
			update(state => ({
				...state,
				activeGraphs: response.graphs.reduce((acc, graph) => {
					acc[graph.workflow_id] = graph;
					return acc;
				}, {} as Record<string, T.ExecutionGraph>),
				selectedWorkflowId: state.selectedWorkflowId ||
					(response.graphs.length > 0 ? response.graphs[0].workflow_id : null)
			}));
		} catch (error) {
			console.error('Failed to fetch active execution graphs:', error);
			update(state => ({ ...state, error: 'Failed to fetch active graphs' }));
		}
	}

	// Select workflow
	function selectWorkflow(workflowId: string | null) {
		update(state => ({ ...state, selectedWorkflowId: workflowId }));
	}

	return {
		subscribe,
		connect,
		disconnect,
		fetchActive,
		selectWorkflow
	};
}

// Export store
export const execution = createExecutionStore();

// Derived stores
export const selectedGraph = derived(execution, ($execution) => {
	if (!$execution.selectedWorkflowId) return null;
	return $execution.activeGraphs[$execution.selectedWorkflowId] || null;
});

export const activeWorkflowIds = derived(execution, ($execution) =>
	Object.keys($execution.activeGraphs)
);

export const hasActiveWorkflows = derived(execution, ($execution) =>
	Object.keys($execution.activeGraphs).length > 0
);
