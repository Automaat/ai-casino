/**
 * Discovery insights store
 */

import { writable } from 'svelte/store';
import type { DiscoveryInsightsResponse } from '$lib/types/api';

export const discoveryInsights = writable<DiscoveryInsightsResponse | null>(null);

const API_BASE =
	typeof window !== 'undefined'
		? `${window.location.protocol}//${window.location.hostname}:8484/api`
		: 'http://localhost:8484/api';

export async function fetchDiscoveryInsights(): Promise<void> {
	try {
		const response = await fetch(`${API_BASE}/discovery/insights`);
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const data = await response.json();
		discoveryInsights.set(data);
	} catch (error) {
		console.error('Failed to fetch discovery insights:', error);
		discoveryInsights.set(null);
	}
}
